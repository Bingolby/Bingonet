//edited by bingo


#pragma once
#include "bingonet/util/bingo_util.h"

namespace bingonet{
	namespace bingo_weight_init{
		class function{
		public:
			virtual void fill(vec_t *weight, cnn_size_t fan_in, cnn_size_t fan_out) = 0;
		};

		class scalable : public function{
		public:
			scalable(float_t value) : scale_(value) {}

			void scale(float_t value) {scale_ = value;}
		protected:
			float_t scale_;
		};

		class xavier : public scalable {
		public:
			xavier() : scalable(float_t(6)) {}
			explicit xavier(float_t value) : scalable(value) {}

			void fill(vec_t* weight, cnn_size_t fan_in, cnn_size_t fan_out) override {
				const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));

				uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
			}
		};

		class constant : public scalable {
		public:
			constant() : scalable(float_t(0)) {}
			explicit constant(float_t value) : scalable(value) {}

			void fill(vec_t* weight, cnn_size_t fan_in, cnn_size_t fan_out){
				std::fill(weight->begin(), weight->end(), scale_);
			}
		};
	}
}