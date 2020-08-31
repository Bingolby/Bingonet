//edited by bingo



#pragma once
#include "bingonet/util/bingo_util.h"

namespace bingonet{
	class bingo_mse{
	public:
		static float_t f(float_t y, float_t t){ return (y - t) * (y - t) / 2;}
		static float_t df(float_t y, float_t t){ return y - t;}
	};

	class bingo_cross_entropy{
	public:
		static float_t f(float_t y, float_t t){
			return -t * std::log(y) - (float_t(1) - t) * std::log(float_t(1) - y);
		}
		static float_t df(float_t y, float_t t){
			return (y - t) / (y * (float_t(1) - y));
		}
	};

	template <typename E>
	vec_t bingo_gradient(const vec_t& y, const vec_t& t){
		vec_t grad(y.size());
		assert(y.size() == t.size());

		for (cnn_size_t i = 0; i < y.size(); i++)
			grad[i] = E::df(y[i], t[i]);

		return grad;
	}

}