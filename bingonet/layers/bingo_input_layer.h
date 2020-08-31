//edited by bingo


#pragma once
#include "bingonet/layers/bingo_layer.h"

namespace bingonet{
class bingo_input_layer : public bingo_layer<bingo_activation::identity>{
public:
	typedef bingo_activation::identity Activation;
	typedef bingo_layer<bingo_activation::identity> Base;
	CNN_USE_BINGO_LAYER_MEMBERS;
	
	bingo_input_layer() : Base(0, 0, 0, 0) {}

	std::string layer_type() const override{ 
		return next_ ? next_->layer_type() : "bingo_input_layer";
	}

	size_t connection_size() const override{ return in_size_;}

	size_t fan_in_size() const override{ return 1;}

	size_t fan_out_size() const override{ return 1;}

	const vec_t& forward_propagation(const vec_t& in, size_t index) override{
		auto& ws = this->get_worker_storage(index);
		ws.output_ = in;
		return next_ ? next_->forward_propagation(in, index) : ws.output_;
	}

	const vec_t& back_propagation(const vec_t& current_delta, size_t index) override{
		return current_delta;
	}

	index3d<cnn_size_t> in_shape() const override{
		return next_ ? next_->in_shape() : index3d<cnn_size_t>(0, 0, 0);
	}

	index3d<cnn_size_t> out_shape() const override{
		return next_ ? next_->out_shape() : index3d<cnn_size_t>(0, 0, 0);
	}

	cnn_size_t in_size() const override{
		return next_ ? next_->in_size() : this->in_size_;
	}

	cnn_size_t out_size() const override{
		return next_ ? next_->out_size() : this->out_size_;
	}
};
}