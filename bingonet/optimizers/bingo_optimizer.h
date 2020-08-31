//edited by bingo

#pragma once
#include "bingonet/util/bingo_util.h"
#include <unordered_map>

namespace bingonet{

	struct bingo_optimizer
	{
		bingo_optimizer() = default;
		bingo_optimizer(const bingo_optimizer& ) = default;
		bingo_optimizer(bingo_optimizer&& ) = default;
		bingo_optimizer& operator = (const bingo_optimizer& ) = default;
		bingo_optimizer& operator = (bingo_optimizer&& ) = default;
		virtual ~bingo_optimizer() = default;
		virtual void reset() {}
	};

	template <int N>
	struct bingo_stateful_optimizer : public bingo_optimizer
	{
		void reset() override {
			for (auto& e : E_) 
				e.clear();
		}

	protected:
		template <int Index>
		vec_t& get(const vec_t& key){
			static_assert(Index < N, "Index out of range");
			if (E_[Index][&key].empty())
				E_[Index][&key].resize(key.size(), float_t());
			return E_[Index][&key];
		}
		std::unordered_map<const vec_t*, vec_t> E_[N];
	};

	struct bingo_adagrad : public bingo_stateful_optimizer<1>
	{
		bingo_adagrad() : alpha(float_t(0.01)), eps(float_t(1e-8)) {}
		void update(const vec_t& dW, vec_t& W){
			vec_t& g = get<0>(W);
			for_i(static_cast<int>(W.size()), [&](int i){
				g[i] += dW[i] * dW[i];
				W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
			});
		}
		float_t alpha;
	private:
		float_t eps;
	};

	struct bingo_RMSprop : public bingo_stateful_optimizer<1>
	{
		bingo_RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)){}
		void update(const vec_t& dW, vec_t& W){
			vec_t& g = get<0>(W);
			for_i(static_cast<int>(W.size()), [&](int i){
				g[i] = mu * g[i] + (1- mu) * dW[i] * dW[i];
				W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
			});
		}

		float_t alpha;
		float_t mu;
	private:
		float_t eps;
	};

	struct bingo_adam : public bingo_stateful_optimizer<2>
	{
		bingo_adam() : alpha(float_t(0.001)), b1(float_t(0.9)), b2(float_t(0.999)), 
		 b1_t(float_t(0.9)), b2_t(float_t(0.999)), eps(float_t(1e-8)){}

		void update(const vec_t& dW, vec_t& W){
			vec_t& mt = get<0>(W);
			vec_t& vt = get<1>(W);

			b1_t *= b1;
			b2_t *= b2;

			for_i(static_cast<int>(W.size()), [&](int i){
				mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
				vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];
				
				W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) / (std::sqrt(vt[i] / (float_t(1) - b2_t)) + eps );
			});
		}
		float_t alpha;
		float_t b1;
		float_t b2;
		float_t b1_t;
		float_t b2_t;
	private:
		float_t eps;
	};

	struct bingo_gradient_descent : public bingo_optimizer
	{
		bingo_gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0)){}
		void update(const vec_t& dW, vec_t& W){
			for_i(static_cast<int>(W.size()), [&](int i){
				W[i] -= alpha * (dW[i] + lambda * W[i]);
			});
		}

		float_t alpha;
		float_t lambda;
	};

	struct bingo_momentum : public bingo_stateful_optimizer<1>{
		bingo_momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}
		void update(const vec_t& dW, vec_t& W){
			vec_t& V = get<0>(W);
			for_i(static_cast<int>(W.size()), [&](int i){
				float_t Vi = mu * V[i] - alpha * (dW[i] + W[i] * lambda);//caffe
				W[i] += Vi;
				V[i] = Vi;
			});
		}
		float_t alpha;
		float_t lambda;
		float_t mu;
	};
}

