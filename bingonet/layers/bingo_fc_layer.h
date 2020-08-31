//edited by bingo


#pragma once
#include "bingonet/layers/bingo_layer.h"
#include "bingonet/util/bingo_product.h"

namespace bingonet{
    template<typename Activation>
    class bingo_fc_layer : public bingo_layer<Activation>{
    public:
    	typedef bingo_layer<Activation> Base;

        CNN_USE_BINGO_LAYER_MEMBERS;

        bingo_fc_layer(cnn_size_t in_dim, cnn_size_t out_dim, bool has_bias = true)
         :  Base(in_dim, out_dim, size_t(in_dim) * size_t(out_dim), has_bias ? out_dim : 0), has_bias_(has_bias) {}

        std::string layer_type() const override { return "bingo_fc_layer"; }
        
        size_t connection_size() const override{
         	return ( size_t(in_size_) * size_t(out_size_) );
         }

        size_t fan_in_size() const override {
        	return (size_t(in_size_));
        }

        size_t fan_out_size() const override{
        	return (size_t(out_size_));
        }

        const vec_t& forward_propagation(const vec_t& in, size_t index) override {
            // std::cout<<index;
        	auto& ws = this->get_worker_storage(index);
        	vec_t& a = ws.a_;
        	vec_t& output = ws.output_;

        	for_i(parallelize_, out_size_, [&](int i){
        		a[i] = float_t(0);
        		for(cnn_size_t j=0;j<in_size_;j++){
        			a[i] += in[j] * W_[j*out_size_ + i];
        		}
        		if(has_bias_)
        			a[i] += b_[i];
        	});
        	for_i(parallelize_, out_size_, [&](int i){
        		output[i] = h_.f(a,i);
        	});

        	return (next_? next_->forward_propagation(output, index) : output);
        }

        const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        	auto& ws = this->get_worker_storage(index);
        	vec_t& prev_delta = ws.prev_delta_;
        	vec_t& dW = ws.dW_;
        	vec_t& db = ws.db_;
        	const vec_t& prev_output = prev_->output(static_cast<int>(index));
        	const bingo_activation::function& prev_h = prev_->activation_function();


            // std::cout<<prev_delta[0];
        	for(cnn_size_t i = 0; i < in_size_; i++){
        		prev_delta[i] = vectorize::dot(&current_delta[0], &W_[i*out_size_], out_size_);
        		prev_delta[i] *= prev_h.df(prev_output[i]);
        	}


        	for_(parallelize_, 0, size_t(out_size_), [&](const blocked_range& r){
                /*if(!index)
                    std::cout<<dW[0];*/
                for(cnn_size_t i=0;i<in_size_;i++){
        		    vectorize::muladd(&current_delta[r.begin()], prev_output[i], r.end() - r.begin(), &dW[i*out_size_ + r.begin()]);
                }
        		if(has_bias_){
        			for(int j=r.begin();j<r.end();j++){
        				db[j] += current_delta[j];
        			}
                }
        		
        	});



        	return prev_->back_propagation(ws.prev_delta_,index);
        }

    protected:
    	bool has_bias_;


    };

}



