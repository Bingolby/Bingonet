//edited by bingo



#pragma once
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <map>
#include <set>

#include "bingonet/util/bingo_util.h"
#include "bingonet/layers/bingo_layers.h"
#include "bingonet/lossfunctions/bingo_loss_function.h"
#include "bingonet/activations/bingo_activation_function.h"



namespace bingonet{
	template<typename LossFunction, typename Optimizer>
	class bingo_network{
	public:
		typedef LossFunction E;
		explicit bingo_network(const std::string& name=""):name_(name) {}

		cnn_size_t in_dim() const{return layers_.head()->in_size();}

		cnn_size_t out_dim() const{return layers_.tail()->out_size();}

		std::string name() const{return name_;}

		Optimizer& optimizer() {return optimizer_;}

		void init_weight() {layers_.init_weight();}

		void add(std::shared_ptr<bingo_layer_base> layer) {layers_.add(layer);}

		vec_t predict(const vec_t& in){return fprop(in);}

		label_t predict_label(const vec_t& in){return fprop_max_index(in);}

		template<typename Range>
		vec_t predict(const Range& in){
			using std::begin;
			using std::end;
			return predict(vec_t(begin(in),end(in)));
		}
		//train without callback
		template <typename T>
		bool train(const std::vector<vec_t>& in,
				   const std::vector<T>&     t,
				   size_t                    batch_size = 1,
				   int                       epoch = 1
			       )
		{
			//set_netphase(net_phase::train);
			return train(in, t, batch_size, epoch, nop, nop);
		}
		//train with callback
		template <typename OnBatchEnumerate, typename OnEpochEnumerate, typename T>
		bool train(const std::vector<vec_t>& in,
				   const std::vector<T>&     t,
				   size_t                    batch_size,
				   int                       epoch,
				   OnBatchEnumerate          on_batch_enumerate,
				   OnEpochEnumerate          on_epoch_enumerate,
				   const bool                reset_weight = true,
				   const int                 n_threads = CNN_TASK_SIZE,
				   const std::vector<vec_t>* t_cost = nullptr
			       )
		{
			check_training_data(in, t);
			check_target_cost_matrix(t, t_cost);//??
			set_netphase(net_phase::train);
			layers_.set_worker_count(n_threads);
			if (reset_weight)
				init_weight();
			layers_.set_parallelize(batch_size < CNN_TASK_SIZE);//?
			optimizer_.reset();

			for(int iter = 0; iter < epoch; iter++){
				for (size_t i=0; i<in.size(); i+=batch_size){
					train_once(&in[i], &t[i],
						       static_cast<int>(std::min(batch_size, in.size() - i)),
						       n_threads,
						       get_target_cost_sample_pointer(t_cost, i));
					on_batch_enumerate();

					if (i % 100 == 0 && layers_.is_exploded()){
						std::cout<< "[Warning]Detected infinite value in weight. stop learning."<<std::endl;
						return false;
					}
				}
				on_epoch_enumerate();
			}
			return true;
		}

		void set_netphase(net_phase phase){
			for (size_t i = 0; i != layers_.depth(); ++i){
				layers_[i]->set_context(phase);
			}
		}


	protected:
		label_t fprop_max_index(const vec_t& in, int idx=0){
			return label_t(max_index(fprop(in,idx)));
		}
	private:
		const vec_t fprop(const vec_t& in, int idx=0){
			if(in.size()!=size_t(in_dim()))
				data_mismatch(*layers_[0], in);
			return layers_.head()->forward_propagation(in,idx);
		}

		void bprop(const vec_t& out, const vec_t& t, int idx, const vec_t* t_cost){
			vec_t tail_delta(out_dim());
			const bingo_activation::function& h = layers_.tail()->activation_function();

			vec_t dE_dy = bingo_gradient<E>(out, t);
			for (size_t i = 0; i<out_dim(); i++){
				tail_delta[i] = dE_dy[i] * h.df(out[i]);
			}

			if (t_cost) {
				for_i(out_dim(), [&](int i){
					tail_delta[i] *= (*t_cost)[i];
				});
			}

			layers_.tail()->back_propagation(tail_delta, idx);
		}

		inline const vec_t* get_target_cost_sample_pointer(const std::vector<vec_t>* t_cost, size_t i) {
        	if (t_cost) {
            	const std::vector<vec_t>& target_cost = *t_cost;
            	assert(i < target_cost.size());
            	return &(target_cost[i]);
        	}
        	else {
            	return nullptr;
        	}
    	}


		std::string name_;
		bingo_layers layers_;
		Optimizer optimizer_;

		template <typename T>
		void check_training_data(const std::vector<vec_t>& in, const std::vector<T>& t){
			cnn_size_t dim_in = in_dim();
			cnn_size_t dim_out = out_dim();

			if(in.size() != t.size()){
				throw nn_error("number of training data must be equal to label data");
			}

			size_t num = in.size();

			for (size_t i=0; i<num; i++){
				if(in[i].size() != dim_in)
					throw nn_error(format_str("input dimension mismatch!\n dim(data[%u])=%d, dim(network input)=%u", i, in[i].size(), dim_in));
				
				check_t(i, t[i], dim_out);
			}
		}

		void check_t(size_t i, const vec_t& t, cnn_size_t dim_out){
			if(t.size() != dim_out)
				throw nn_error(format_str("output dimension mismatch!\n dim(target[%u])=%u, dim(network output size=%u", i, t.size(), dim_out));
		}

		template <typename T>
		void check_target_cost_matrix(const std::vector<T>& t, const std::vector<vec_t>* t_cost) {
			if(t_cost != nullptr){
				if(t.size() != t_cost->size()){
					throw nn_error("if target cost is supplied, its length must equal that of target data");
				}

				for (size_t i=0, end = t.size(); i<end; i++){
					check_target_cost_element(t[i], t_cost->operator[](i));
				}
			}
		}
		//classification
		void check_target_cost_element(const label_t t, const vec_t& t_cost){
			if(t >= t_cost.size()){
				throw nn_error("if target cost is supplied for a classification task, some cost must be given for each distinct class label");
			}
		}
		//regression
		void check_target_cost_element(const vec_t& t, const vec_t& t_cost){
			if(t.size() != t_cost.size()){
				throw nn_error("if target cost is supplied for a regression task, its shape must be identical to the target data");
			}
		}

		float_t target_value_min() const { return layers_.tail()->activation_function().scale().first; }
		float_t target_value_max() const { return layers_.tail()->activation_function().scale().second; }

		void label2vector(const label_t* t, int num, std::vector<vec_t> *vec) const{
			cnn_size_t outdim = out_dim();

			assert(num > 0);
			assert(outdim > 0);

			vec->reserve(num);
			for (int i = 0; i < num; i++){
				assert(t[i] < outdim);
				vec->emplace_back(outdim, target_value_min());
				vec->back()[t[i]] = target_value_max();
			}
		}

		//label->target_output
		void train_once(const vec_t* in, const label_t* t, int size, const int n_threads, const vec_t* t_cost) {
			std::vector<vec_t> v;
			label2vector(t, size, &v);
			train_once(in, &v[0], size, n_threads, t_cost);
		}

		void train_once(const vec_t* in, const vec_t* t, int size, const int n_threads, const vec_t* t_cost) {
			if (size == 1){
				bprop(fprop(in[0]), t[0], 0, t_cost);
				layers_.update_weights(&optimizer_, 1, 1);
			} else{
				train_onebatch(in, t, size, n_threads, t_cost);
			}
		}

		void train_onebatch(const vec_t* in, const vec_t* t, int batch_size, const int num_tasks, const vec_t* t_cost){
			int num_threads = std::min(batch_size, num_tasks);

			int data_per_thread = (batch_size + num_threads -1) / num_threads;

			for_i(num_threads, [&](int i){
				int start_index = i * data_per_thread;
				int end_index = std::min(batch_size, start_index + data_per_thread);

				for(int j = start_index; j < end_index; ++j)
					bprop(fprop(in[j], i), t[j], i, t_cost ? &(t_cost[j]) : nullptr);
			}, 1);

			layers_.update_weights(&optimizer_, num_threads, batch_size);
		}
	};

	template <typename L, typename O, typename Layer>
	bingo_network<L, O>& operator << (bingo_network<L, O>& nn, Layer&& l){
		nn.add(std::make_shared<typename std::remove_reference<Layer>::type>(std::forward<Layer>(l)));
		return nn;
	}
}//namespace bingonet
