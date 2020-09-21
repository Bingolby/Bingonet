//edited by bingo




#pragma once
#include <sstream>
#include <iomanip>
#include <memory>
#include <stdio.h>
#include "bingonet/util/bingo_util.h"
#include "bingonet/util/bingo_product.h"
#include "bingonet/util/bingo_weight_init.h"
#include "bingonet/activations/bingo_activation_function.h"


namespace bingonet{

	class bingo_layer_base{
	public:
		friend void connection_mismatch(const bingo_layer_base& from, const bingo_layer_base& to);

		bingo_layer_base(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim)
			:parallelize_(true), next_(nullptr), prev_(nullptr),
			 weight_init_(std::make_shared<bingo_weight_init::xavier>()),
			 bias_init_(std::make_shared<bingo_weight_init::constant>(float_t(0)))
		{
			set_size(in_dim, out_dim, weight_dim, bias_dim);
			set_worker_count(CNN_TASK_SIZE);
		}

		bingo_layer_base(const bingo_layer_base&) = default;

		vec_t& weight() { return W_; }
		vec_t& bias() { return b_; }

		virtual ~bingo_layer_base() = default;

		virtual std::string layer_type() const = 0;

		virtual size_t connection_size() const = 0;

		virtual size_t fan_in_size() const = 0;

		virtual size_t fan_out_size() const = 0;

		virtual const vec_t& forward_propagation(const vec_t& in, size_t worker_index) = 0;

		virtual const vec_t& back_propagation(const vec_t& in, size_t worker_index) = 0;

		virtual void set_worker_count(cnn_size_t worker_count){
			if(worker_count == 0)
				throw nn_error("worker_count cannot be zero");

			if(worker_count != worker_storage_.size()){
				worker_storage_.resize(worker_count);
				set_size(in_size_, out_size_, W_.size(), b_.size());
			}
		}

		const vec_t output(cnn_size_t worker_index) const {
			return worker_storage_[worker_index].output_;
		}

		virtual bingo_activation::function& activation_function() = 0;
		
		void connect(std::shared_ptr<bingo_layer_base>& tail){
			if(out_size() != 0 && tail->in_size() != out_size())
				connection_mismatch(*this, *tail);
			next_ = tail.get();
			tail->prev_ = this;
		}

		virtual cnn_size_t in_size() const{return in_size_;}

		virtual cnn_size_t out_size() const{return out_size_;}

		virtual index3d<cnn_size_t> in_shape() const{return index3d<cnn_size_t>(in_size(), 1, 1);}

		virtual index3d<cnn_size_t> out_shape() const{return index3d<cnn_size_t>(out_size(), 1, 1);}

		void set_parallelize(bool parallelize){
			parallelize_ = parallelize;
		}

		void init_weight(){
			weight_init_->fill(&W_, static_cast<cnn_size_t>(fan_in_size()),
									static_cast<cnn_size_t>(fan_out_size()));
			bias_init_->fill(&b_, static_cast<cnn_size_t>(fan_in_size()),
								  static_cast<cnn_size_t>(fan_out_size()));
			clear_diff();
		}

		template <typename Optimizer>
		void update_weight(Optimizer* o, cnn_size_t worker_size, cnn_size_t batch_size){
			if (W_.empty()) return;

			merge(worker_size, batch_size);

			o->update(worker_storage_[0].dW_, W_);
			o->update(worker_storage_[0].db_, b_);

			clear_diff();
		}

		bool is_exploded() const{return has_infinite(W_) || has_infinite(b_);}

		virtual void set_context(net_phase ctx) { }

	protected:
		cnn_size_t in_size_;
		cnn_size_t out_size_;
		bool parallelize_;

		bingo_layer_base* next_;
		bingo_layer_base* prev_;
		vec_t W_;
		vec_t b_;
		
		struct worker_specific_storage{
			vec_t a_;
			vec_t output_;
			vec_t prev_delta_;
			
			vec_t dW_;
			vec_t db_;
		};		

		std::shared_ptr<bingo_weight_init::function> weight_init_;
		std::shared_ptr<bingo_weight_init::function> bias_init_;

		const worker_specific_storage& get_worker_storage(cnn_size_t worker_index) const{
			return worker_storage_[worker_index];
		}

		worker_specific_storage& get_worker_storage(cnn_size_t worker_index) {
			return worker_storage_[worker_index];
		}

	private:
		std::vector<worker_specific_storage> worker_storage_;

		void set_size(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim) {
			in_size_ = in_dim;
			out_size_ = out_dim;

			W_.resize(weight_dim);
			b_.resize(bias_dim);

			for (auto& ws:worker_storage_){
				ws.a_.resize(out_dim);
				ws.output_.resize(out_dim);
				ws.prev_delta_.resize(in_dim);

				ws.dW_.resize(weight_dim);
				ws.db_.resize(bias_dim);
			}
		}

		void merge(cnn_size_t worker_size, cnn_size_t batch_size){
			auto& ws = worker_storage_;

			for (cnn_size_t i = 1; i< worker_size; i++){
				vectorize::reduce<float_t>(&ws[i].dW_[0],
					static_cast<cnn_size_t>(ws[i].dW_.size()), &ws[0].dW_[0]);
			}
			for (cnn_size_t i = 1; i< worker_size; i++){
				vectorize::reduce<float_t>(&ws[i].db_[0],
					static_cast<cnn_size_t>(ws[i].db_.size()), &ws[0].db_[0]);
			}

			std::transform(ws[0].dW_.begin(), ws[0].dW_.end(), ws[0].dW_.begin(), [&](float_t x){
				return x / batch_size;
			});
			std::transform(ws[0].db_.begin(), ws[0].db_.end(), ws[0].db_.begin(), [&](float_t x){
				return x / batch_size;
			});
		}

		void clear_diff(){
			for (auto& ws : worker_storage_){
				std::fill(ws.dW_.begin(), ws.dW_.end(), float_t(0));
				std::fill(ws.db_.begin(), ws.db_.end(), float_t(0));
			}
		}
	};

	template<typename Activation>
	class bingo_layer : public bingo_layer_base{
	public:
		bingo_layer(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim)
			: bingo_layer_base(in_dim, out_dim, weight_dim, bias_dim) {}

		bingo_activation::function& activation_function() override { return h_; }
	protected:
		Activation h_;
	};

	inline void connection_mismatch(const bingo_layer_base& from, const bingo_layer_base& to) {
	    std::ostringstream os;

	    os << std::endl;
	    os << "output size of Nth layer must be equal to input of (N+1)th layer" << std::endl;
	    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:" << from.in_size() << "(" << from.in_shape() << "), " << 
	                                                "out:" << from.out_size() << "(" << from.out_shape() << ")" << std::endl;
	    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:" << to.in_size() << "(" << to.in_shape() << "), " <<
	                                             "out:" << to.out_size() << "(" << to.out_shape() << ")" << std::endl;
	    os << from.out_size() << " != " << to.in_size() << std::endl;
	    std::string detail_info = os.str();

	    throw nn_error("layer dimension mismatch!" + detail_info);
	}

	inline void data_mismatch(const bingo_layer_base& layer, const vec_t& data) {
    	 std::ostringstream os;

    	os << std::endl;
    	os << "data dimension:    " << data.size() << std::endl;
    	os << "network dimension: " << layer.in_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;

    	std::string detail_info = os.str();

    	throw nn_error("input dimension mismath!" + detail_info);
 	}
}