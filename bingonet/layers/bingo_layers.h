//edited by bingo


#pragma once
#include "bingonet/layers/bingo_layer.h"
#include "bingo_input_layer.h"

namespace bingonet{
	class bingo_layers{
	public:
		bingo_layers(){ add(std::make_shared<bingo_input_layer>());}

		bingo_layers(const bingo_layers& rhs) { construct(rhs);}

		bingo_layers& operator = (const bingo_layers& rhs) {
			layers_.clear();
			construct(rhs);
			return *this;
		}

		void add(std::shared_ptr<bingo_layer_base> new_tail){
			if(tail())
				tail()->connect(new_tail);
			layers_.push_back(new_tail);
		}

		bingo_layer_base* head() const{ return is_empty() ? 0 : layers_[0].get();}

		bingo_layer_base* tail() const{ return is_empty() ? 0 : layers_[layers_.size()-1].get();}

		bool is_empty() const { return layers_.size() == 0;}
		
		void set_worker_count(cnn_size_t thread_count){
			for (auto pl : layers_){
				pl->set_worker_count(thread_count);
			}
		}

		void init_weight(){
			for (auto pl: layers_)
				pl->init_weight();
		}

		void set_parallelize(bool set_parallelize){
			for (auto pl : layers_)
				pl->set_parallelize(set_parallelize);
		}

		template <typename Optimizer>
		void update_weights(Optimizer* o, size_t worker_size, size_t batch_size){
			for (auto pl : layers_)
				pl->update_weight(o,static_cast<cnn_size_t>(worker_size), batch_size);
		}

		size_t depth() const{
			return layers_.size() - 1;
		}

		const bingo_layer_base* operator[](size_t index) const{
			return layers_[index + 1].get();
		}

		bingo_layer_base* operator[](size_t index) {
			return layers_[index + 1].get();
		}

		bool is_exploded() const{
			for (auto pl : layers_)
				if(pl->is_exploded()) return true;
			return false;
		}

	private:
		std::vector<std::shared_ptr<bingo_layer_base>> layers_;

		void construct(const bingo_layers& rhs){
			add(std::make_shared<bingo_input_layer>());
			for (size_t i=1;i<rhs.layers_.size();i++){
				add(rhs.layers_[i-1]);
			}
		}
	};
}