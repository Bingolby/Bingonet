//edited by bingo


#pragma once
#include "bingonet/layers/bingo_layer.h"
#include "bingonet/util/bingo_product.h"

namespace bingonet{
    template<typename Activation>
    class bingo_conv_layer : public bingo_layer<Activation>{
    public:
    	typedef bingo_layer<Activation> Base;

        CNN_USE_BINGO_LAYER_MEMBERS;

        bingo_conv_layer(cnn_size_t in_channels, cnn_size_t out_channels, cnn_size_t window_size, cnn_size_t in_width, cnn_size_t in_height, bool has_bias = true)
         :  Base(in_channels * in_width * in_height,
         		 out_channels * (in_width - window_size + 1) * (in_height - window_size + 1),
         		 window_size * window_size * in_channels * out_channels,
         		 has_bias ? out_channels : 0),
         	in_channels_(in_channels), out_channels_(out_channels),
         	window_size_(window_size),
         	in_width_(in_width), in_height_(in_height),
         	has_bias_(has_bias),
         	out_width_(in_width - window_size + 1), out_height_(in_height - window_size + 1){}

        std::string layer_type() const override { return "bingo_conv_layer"; }
        
        size_t connection_size() const override{
         	return size_t(in_size_ * fan_out_size());
         }

        size_t fan_in_size() const override {
        	return size_t(window_size_ * window_size_ * in_channels_);
        }

        size_t fan_out_size() const override{
        	return size_t(window_size_ * window_size_ * out_channels_);
        }

        const vec_t& forward_propagation(const vec_t& in, size_t index) override {
            // std::cout<<index;
        	auto& ws = this->get_worker_storage(index);
        	vec_t& a = ws.a_;
        	vec_t& output = ws.output_;

        	for_i(parallelize_, out_channels_, [&](int o){
        		for(cnn_size_t i = 0; i < in_channels_; i++){
        			const float_t* pw = &W_[(o * in_channels_ + i) * window_size_ * window_size_];
        			const float_t* pi = &in[i * in_height_ * in_width_];
        			float_t* pa = &a[o * out_height_ * out_width_];

        			for(cnn_size_t y = 0; y < out_height_; y++){
        				for(cnn_size_t x = 0; x < out_width_; x++){
        					const float_t* ppw = pw;
        					const float_t* ppi = pi + y * in_width_ + x;
        					float_t sum(0);

        					for(cnn_size_t wy = 0; wy < window_size_; wy++){
        						for(cnn_size_t wx = 0; wx < window_size_; wx++){
        							sum += (*ppw) * ppi[wy * in_width_ + wx];
        							++ppw;
        						}
        					}
        					pa[y * out_width_ + x] += sum;
        				}
        			}
        		}
        		if(has_bias_){
        			float_t* pa = &a[o * out_width_ * out_height_];
        			float_t b = b_[o];
        			std::for_each(pa, pa + out_width_ * out_height_, [&](float_t& x){x += b;});
        		}
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

        	for_i(parallelize_, in_channels_, [&](int i){
        		for(cnn_size_t o = 0; o < out_channels_; o++){
        			const float_t* pw = &W_[(o * in_channels_ + i) * window_size_ * window_size_];
        			const float_t* p_src = &current_delta[o * out_height_ * out_width_];
        			float_t* p_dst = &prev_delta[i * in_width_ * in_height_];

        			for(cnn_size_t y = 0; y < out_height_; y++){
        				for(cnn_size_t x = 0; x < out_width_; x++){
        					const float_t* ppw = pw;
        					const float_t* pp_src = p_src + y * out_width_ + x;
        					float_t* pp_dst = p_dst + y * in_width_ + x;

        					for(cnn_size_t wy = 0; wy < window_size_; wy++){
        						for(cnn_size_t wx = 0; wx < window_size_; wx++){
        							pp_dst[wy * in_width_ + wx] += (*pp_src) * (*ppw);
        							++ppw;
        						}
        					}
        				}
        			}
        		}
        	});
        	for_i(parallelize_, in_size_, [&](int i){
        		prev_delta[i] *= prev_h.df(prev_output[i]);
        	});

        	for_i(parallelize_, out_channels_, [&](int o){
        		for(cnn_size_t i = 0; i < in_channels_; i++){

        			for(cnn_size_t wy = 0; wy < window_size_; wy++){
        				for(cnn_size_t wx = 0; wx < window_size_; wx++){
        					const float_t* p_prev_out = &prev_output[i * in_width_ * in_height_ + wy * in_width_ + wx];
        					const float_t* p_curr_delta = &current_delta[o * out_width_ * out_height_];
        					float_t* p_dW = &dW[(o * in_channels_ + i) * window_size_ * window_size_ + wy * window_size_ + wx];

        					for(cnn_size_t y = 0; y < out_height_; y++){
        						*p_dW += vectorize::dot(p_curr_delta + y * out_width_, p_prev_out + y * in_width_, out_width_);
        					}
        				}
        			}
        		}

        		if(has_bias_){
        			const float_t* p_curr_delta = &current_delta[o * out_width_ * out_height_];
        			db[o] += std::accumulate(p_curr_delta, p_curr_delta + out_width_ * out_height_, float_t(0));
        		}
        	});

        	return prev_->back_propagation(ws.prev_delta_,index);
        }



    protected:
    	cnn_size_t in_channels_;
    	cnn_size_t out_channels_;
    	cnn_size_t window_size_;
    	cnn_size_t in_width_;
    	cnn_size_t in_height_;
    	cnn_size_t out_width_;
    	cnn_size_t out_height_;
    	bool has_bias_;
    };
}



