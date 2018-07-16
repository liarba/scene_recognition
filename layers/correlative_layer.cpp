//correlative embeding learning loss
//revised based on lifted struture loss https://github.com/rksltnl/Deep-Metric-Learning-CVPR16

#include <algorithm>
#include <vector>

#include "caffe/layers/correlative_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CorrelativeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  // List of member variables defined in /include/caffe/loss_layers.hpp;
  //   diff_, dist_sq_, summer_vec_, loss_aug_inference_;
  if (this->layer_param_.loss_param().has_weight_source()) {
    const string& weight_source = this->layer_param_.loss_param().weight_source();
    LOG(INFO) << "Opening file " << weight_source;
    std::fstream infile(weight_source.c_str(), std::fstream::in);
    CHECK(infile.is_open());
    Dtype tmp_val;
    while (infile >> tmp_val) {
      CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
      loss_weights_.push_back(tmp_val);
    }
    
    infile.close();
    int dim = this->layer_param_.correlative_loss_param().dim();
    CHECK_EQ(loss_weights_.size(), dim);
  } else {
    LOG(INFO) << "Weight_Loss file is not provided. Assign all one to it."; // in practice we do not use loss weights
    int dim1 = this->layer_param_.correlative_loss_param().dim();
    loss_weights_.assign(dim1, 1.0);
  }
  
  dist_sq_1.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_2.Reshape(bottom[1]->num(), 1, 1, 1);
  dot_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  ones_.Reshape(bottom[0]->num(), 1, 1, 1);  // n by 1 vector of ones.
  for (int i=0; i < bottom[0]->num(); ++i){
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
  blob_pos_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
  blob_neg_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
} 

template <typename Dtype>
void CorrelativeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); i++){
    dist_sq_1.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (i*channels));
  }
  for (int i = 0; i < bottom[0]->num(); i++){
    dist_sq_2.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[1]->cpu_data() + (i*channels), bottom[1]->cpu_data() + (i*channels));
  }

  int M_ = bottom[0]->num();
  int N_ = bottom[0]->num();
  int K_ = bottom[0]->channels();

  const Dtype* bottom_data1 = bottom[0]->cpu_data(); //modal 1
  const Dtype* bottom_data2 = bottom[1]->cpu_data(); //modal 2

  Dtype dot_scaler(-2.0);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, dot_scaler, bottom_data1, bottom_data2, (Dtype)0., dot_.mutable_cpu_data());

  for (int i=0; i<N_; i++){
    caffe_axpy(N_, dist_sq_1.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  for (int i=0; i<N_; i++){
    caffe_axpy(N_, Dtype(1.0), dist_sq_2.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  vector<vector<bool> > label_mat(N_, vector<bool>(N_, false));
  for (int i=0; i<N_; i++){
    for (int j=0; j<N_; j++){
      label_mat[i][j] = (bottom[2]->cpu_data()[i] == bottom[2]->cpu_data()[j]);
    }
  }

  Dtype margin = this->layer_param_.correlative_loss_param().margin();
  Dtype loss(0.0);
  num_constraints = Dtype(0.0); 
  const Dtype* bin1 = bottom[0]->cpu_data();
  Dtype* bout1 = bottom[0]->mutable_cpu_diff();
  const Dtype* bin2 = bottom[1]->cpu_data();
  Dtype* bout2 = bottom[1]->mutable_cpu_diff();
  for (int i=0; i<N_; i++){
    caffe_set(K_, Dtype(0.0), bout1 + i*K_);
  }
  for (int i=0; i<N_; i++){
    caffe_set(K_, Dtype(0.0), bout2 + i*K_);
  }
  for (int i=0; i<N_; i++){
    
      // take modal 1 (r_i) as abchor positive 
      // a positive pair (ri, di)
      if (label_mat[i][i]){
        Dtype dist_pos = sqrt(dot_.cpu_data()[i*N_ + i]);

        caffe_sub(K_, bin1 + i*K_, bin2 + i*K_, blob_pos_diff_.mutable_cpu_data());

        // 1.count the number of negatives for this positive
        int num_negatives = 0;
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            num_negatives += 1;
          }
        }

        loss_aug_inference_.Reshape(num_negatives, 1, 1, 1);

        // vector of ones used to sum along channels
        summer_vec_.Reshape(num_negatives, 1, 1, 1);
        for (int ss = 0; ss < num_negatives; ++ss){
          summer_vec_.mutable_cpu_data()[ss] = Dtype(1);
        }

        // 2. compute loss augmented inference
        int neg_idx = 0;
        // mine negative (anchor ri, neg dk)
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            loss_aug_inference_.mutable_cpu_data()[neg_idx] = margin - sqrt(dot_.cpu_data()[i*N_ + k]);
            neg_idx++;
          }
        }

        // compute softmax of loss aug inference vector;
        Dtype max_elem = *std::max_element(loss_aug_inference_.cpu_data(), loss_aug_inference_.cpu_data() + num_negatives);

        caffe_add_scalar(loss_aug_inference_.count(), Dtype(-1.0)*max_elem, loss_aug_inference_.mutable_cpu_data());
        caffe_exp(loss_aug_inference_.count(), loss_aug_inference_.mutable_cpu_data(), loss_aug_inference_.mutable_cpu_data());
        Dtype soft_maximum = log(caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data())) + max_elem;
        Dtype loss_weight_label=loss_weights_[bottom[2]->cpu_data()[i]];
        // hinge the soft_maximum - S_ridi (positive pair similarity)
        Dtype this_loss = std::max(soft_maximum + dist_pos, Dtype(0.0));

        // squared hinge
        loss += loss_weight_label*this_loss * this_loss; 
        num_constraints += loss_weight_label;

        // 3. compute gradients
        Dtype sum_exp = caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data());

        Dtype scaler(0.0);

        scaler = loss_weight_label*Dtype(2.0)*this_loss / dist_pos;
        // update r_i
        caffe_axpy(K_, scaler * Dtype(1.0), blob_pos_diff_.cpu_data(), bout1 + i*K_);
        // update d_i
        caffe_axpy(K_, scaler * Dtype(-1.0), blob_pos_diff_.cpu_data(), bout2 + i*K_);

        neg_idx = 0;
        Dtype dJ_dDik(0.0);
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            caffe_sub(K_, bin1 + i*K_, bin2 + k*K_, blob_neg_diff_.mutable_cpu_data());

            dJ_dDik = Dtype(2.0)*this_loss * Dtype(-1.0)* loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
            neg_idx++;

            scaler = loss_weight_label*dJ_dDik / sqrt(dot_.cpu_data()[i*N_ + k]);

            // update r_i
            caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout1 + i*K_);
            // update d_k
            caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout2 + k*K_);
          }
        }

      } // close this postive pair
    
  }
  //loop
  for (int i=0; i<N_; i++){
    
      // take modal 2 (d_i) as abchor positive 
      // found a positive pair(d_i, r_i)
      if (label_mat[i][i]){
        Dtype dist_pos = sqrt(dot_.cpu_data()[i*N_ + i]);

        caffe_sub(K_, bin2 + i*K_, bin1 + i*K_, blob_pos_diff_.mutable_cpu_data());

        // 1.count the number of negatives for this positive
        int num_negatives = 0;
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            num_negatives += 1;
          }
        }

        loss_aug_inference_.Reshape(num_negatives, 1, 1, 1);

        // vector of ones used to sum along channels
        summer_vec_.Reshape(num_negatives, 1, 1, 1);
        for (int ss = 0; ss < num_negatives; ++ss){
          summer_vec_.mutable_cpu_data()[ss] = Dtype(1);
        }

        // 2. compute loss augmented inference
        int neg_idx = 0;
        // mine negative (d_i, r_k)
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            loss_aug_inference_.mutable_cpu_data()[neg_idx] = margin - sqrt(dot_.cpu_data()[k*N_ + i]);
            neg_idx++;
          }
        }

        // compute softmax of loss aug inference vector;
        Dtype max_elem = *std::max_element(loss_aug_inference_.cpu_data(), loss_aug_inference_.cpu_data() + num_negatives);

        caffe_add_scalar(loss_aug_inference_.count(), Dtype(-1.0)*max_elem, loss_aug_inference_.mutable_cpu_data());
        caffe_exp(loss_aug_inference_.count(), loss_aug_inference_.mutable_cpu_data(), loss_aug_inference_.mutable_cpu_data());
        Dtype soft_maximum = log(caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data())) + max_elem;

        // hinge the soft_maximum - S_diri (positive pair similarity)
        Dtype this_loss2 = std::max(soft_maximum + dist_pos, Dtype(0.0));
        Dtype loss_weight_label2=loss_weights_[bottom[2]->cpu_data()[i]];
        // squared hinge
        loss += loss_weight_label2*this_loss2 * this_loss2; 
        num_constraints += loss_weight_label2;

        // 3. compute gradients
        Dtype sum_exp = caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data());

        Dtype scaler(0.0);

        scaler = loss_weight_label2*Dtype(2.0)*this_loss2 / dist_pos;
        // update d_i
        caffe_axpy(K_, scaler * Dtype(1.0), blob_pos_diff_.cpu_data(), bout2 + i*K_);
        // update r_i
        caffe_axpy(K_, scaler * Dtype(-1.0), blob_pos_diff_.cpu_data(), bout1 + i*K_);

  
        neg_idx = 0;
        Dtype dJ_dDik(0.0);
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            caffe_sub(K_, bin2 + i*K_, bin1 + k*K_, blob_neg_diff_.mutable_cpu_data());

            dJ_dDik = Dtype(2.0)*this_loss2 * Dtype(-1.0)* loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
            neg_idx++;

            scaler = loss_weight_label2*dJ_dDik / sqrt(dot_.cpu_data()[k*N_ + i]);

            // update d_i
            caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout2 + i*K_);
            // update r_k
            caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout1 + k*K_);
          }
        }

      } // close this postive pair
    
  }
  loss = loss / num_constraints / Dtype(2.0);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CorrelativeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype alpha = top[0]->cpu_diff()[0] / num_constraints / Dtype(2.0);

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for (int i = 0; i < num; i++){
    Dtype* bout1 = bottom[0]->mutable_cpu_diff();
    Dtype* bout2 = bottom[1]->mutable_cpu_diff();
    caffe_scal(channels, alpha, bout1 + (i*channels));
    caffe_scal(channels, alpha, bout2 + (i*channels));
  }
}

#ifdef CPU_ONLY
STUB_GPU(CorrelativeLossLayer);
#endif

INSTANTIATE_CLASS(CorrelativeLossLayer);
REGISTER_LAYER_CLASS(CorrelativeLoss);

}  // namespace caffe



