#ifndef CAFFE_CORRELATIVE_LAYER_HPP_
#define CAFFE_CORRELATIVE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CorrelativeLossLayer : public LossLayer<Dtype> {
 public:
  explicit CorrelativeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "CorrelativeLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dist_sq_1; 
  Blob<Dtype> dist_sq_2;  // cached for backward pass
  Blob<Dtype> dot_;
  Blob<Dtype> ones_;
  Blob<Dtype> blob_pos_diff_;
  Blob<Dtype> blob_neg_diff_;
  Blob<Dtype> loss_aug_inference_;
  Blob<Dtype> summer_vec_;
  Dtype num_constraints;
  std::vector<Dtype> loss_weights_;
};

}  // namespace caffe

#endif  
