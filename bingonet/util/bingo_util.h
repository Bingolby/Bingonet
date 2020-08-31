//edited by bingo



#pragma once
#include <vector>
#include <functional>
#include <random>
#include <type_traits>
#include <limits>
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <sstream>
#include "nn_error.h"
#include "bingonet/config.h"
#include "bingo_aligned_allocator.h"

#ifdef CNN_USE_TBB
#include <tbb/tbb.h>
#include <tbb/task_group.h>
#endif

#ifndef CNN_USE_OMP
#include <thread>
#include <future>
#endif


namespace bingonet{
	typedef cnn_size_t label_t;

	typedef cnn_size_t layer_size_t;
	
	typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;

	template <typename T>
	struct index3d;

	typedef index3d<cnn_size_t> layer_shape_t;

	enum class net_phase{
		train,
		test
	};

	#define CNN_USE_BINGO_LAYER_MEMBERS using bingo_layer_base::in_size_;\
	    using bingo_layer_base::out_size_; \
	    using bingo_layer_base::parallelize_; \
	    using bingo_layer_base::next_; \
	    using bingo_layer_base::prev_; \
	    using bingo_layer_base::W_; \
	    using bingo_layer_base::b_; \
	    using bingo_layer<Activation>::h_

	inline void nop() {
		//do nothing
	}

	template<typename T> inline
	typename std::enable_if<std::is_integral<T>::value, T>::type
	uniform_rand(T min, T max){
		static std::mt19937 gen(1);
		std::uniform_int_distribution<T> dst(min, max);
		return dst(gen);
	}

	template<typename T> inline
	typename std::enable_if<std::is_floating_point<T>::value, T>::type
	uniform_rand(T min, T max){
		static std::mt19937 gen(1);
		std::uniform_real_distribution<T> dst(min, max);
		return dst(gen);
	}

	template<typename T> inline
	typename std::enable_if<std::is_floating_point<T>::value, T>::type
	gaussian_rand(T mean, T sigma){
		static std::mt19937 gen(1);
		std::normal_distribution<T> dst(mean, sigma);
		return dst(gen);
	}

	template<typename Iter>
	void uniform_rand(Iter begin, Iter end, float_t min, float_t max){
		for (Iter it = begin; it != end; ++it)
			*it = uniform_rand(min, max);
	}

	template<typename Iter>
	void gaussian_rand(Iter begin, Iter end, float_t mean, float_t sigma){
		for (Iter it = begin; it != end; ++it)
			*it = gaussian_rand(mean, sigma);
	}

	template<typename T>
	size_t max_index(const T& vec){
		auto begin_iterator = std::begin(vec);
		return std::max_element(begin_iterator, std::end(vec) - begin_iterator);
	}

	template <typename T> 
	inline T sqr(T value){ return value*value;}

	inline bool isfinite(float_t x){
		return x == x;
	}

	template <typename Container>
	inline bool has_infinite(const Container& c){
		for(auto v : c)
			if(!isfinite(v)) return true;
		return false;
	}

	#ifdef CNN_USE_TBB

	static tbb::task_scheduler_init tbbScheduler(tbb::task_scheduler_init::automatic);

	typedef tbb::blocked_range<int> blocked_range;

	template <typename Func>
	void parallel_for(int begin, int end, const Func& f, int grainsize){
		tbb::parallel_for(blocked_range(begin, end, 
			end - begin > grainsize ? grainsize : 1), f);
	}

	template<typename Func>
	void xparallel_for(int begin, int end, const Func& f){
		f(blocked_range(begin, end, 100));
	}

	#else

	struct blocked_range{
		typedef int const_iterator;

		blocked_range(int begin, int end) : begin_(begin), end_(end) {}
		blocked_range(size_t begin, size_t end) : begin_(static_cast<int>(begin)), 
												  end_(static_cast<int>(end)){}
		const_iterator begin() const{ return begin_;}
		const_iterator end() const{ return end_;}

	private:
		int begin_;
		int end_;
	};

	template <typename Func>
	void xparallel_for(size_t begin, size_t end, const Func& f){
		blocked_range r(begin, end);
		f(r);
	}

	#ifdef CNN_USE_OMP
	//a = grainsize
	template <typename Func>
	void parallel_for(int begin, int end, const Func& f, int a){
		#pragma omp parallel for
		for(int i=begin; i<end; ++i)
			f(blocked_range(i,i+1));
	}

	#else
	//a = grainsize
	template <typename Func>
	void parallel_for(int start, int end, const Func& f, int a){
		int hardware_threads = std::thread::hardware_concurrency();
		int nthreads = (hardware_threads != 0 ? hardware_threads : 2);
		int blockSize = (end - start) / nthreads;
		if (blockSize*nthreads < end - start) //really need?
			blockSize++;

		std::vector<std::future<void>> futures;

		int blockStart = start;
		int blockEnd = blockStart + blockSize;
		if(blockEnd > end) blockEnd = end;

		for(int i = 0; i < nthreads; i++){
			futures.push_back(std::move(std::async(std::launch::async, [blockStart, blockEnd, &f]{
				f(blocked_range(blockStart, blockEnd));
			})));

			blockStart += blockSize;
			blockEnd = blockStart + blockSize;
			if(blockStart >= end) break;
			if(blockEnd > end) blockEnd = end;
		}
		for(auto &future : futures)
			future.wait();
	}

	#endif//CNN_USE_OMP
	#endif//CNN_USE_TBB

	template <typename T, typename Func>
	void for_i(T size, Func f, int grainsize = 100){
		for_i(true, size, f, grainsize);
	}

	template <typename T, typename Func>
	inline
	void for_i(bool parallelize, T size, Func f, int grainsize = 100){
		for_(parallelize, 0, size, [&](const blocked_range& r){
			#ifdef CNN_USE_OMP
			#pragma omp parallel for
			#endif
			for(int i = r.begin(); i<r.end(); i++)
				f(i);
		}, grainsize);
	}

	template <typename T, typename Func>
	inline
	void for_(bool parallelize, int begin, T end, Func f, int grainsize = 100){
		static_assert(std::is_integral<T>::value, "end must be integral type");
		for_(typename std::is_unsigned<T>::type(), parallelize, begin, end, f, grainsize);
	}

	template <typename T, typename U>//always true?
	bool value_representation(U const& value){
		return static_cast<U>(static_cast<T>(value)) == value;
	}

	template <typename T, typename Func>
	inline
	void for_(std::true_type, bool parallelize, int begin, T end, Func f, int grainsize = 100){
		parallelize = parallelize && value_representation<int>(end);
		parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize) : 
					  xparallel_for(begin, static_cast<int>(end), f);	
	}

	template <typename T, typename Func>
	inline
	void for_(std::false_type, bool parallelize, int begin, T end, Func f, int grainsize = 100){
		parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize) : 
					  xparallel_for(begin, static_cast<int>(end), f);
	}

	

	inline std::string format_str(const char *fmt, ...){
		static char buf[2048];
		va_list args;
		va_start(args, fmt);
		vsnprintf(buf, sizeof(buf), fmt, args);
		va_end(args);
		return std::string(buf);
	}

	template <typename T>
	struct index3d{
		index3d(T width, T height, T depth){
			reshape(width, height, depth);
		}

		index3d() : width_(0), height_(0), depth_(0){}

		void reshape(T width, T height, T depth){
			width_ = width;
			height_ = height;
			depth_ = depth;

			if((long long) width * height * depth > std::numeric_limits<T>::max())
				throw nn_error(
					format_str("error while constructing layer: layer size too large for BingoNet\nWidthxHeightxChannels=%dx%dx%d >= max size of [%s](=%d)",
						width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
		}

		T get_index(T x, T y, T channel) const{
			assert(x >= 0 && x < width_);
			assert(y >= 0 && y < height_);
			assert(channel >= 0 && channel < depth_);
			return (height_ * channel + y) * width_ + x;
		}

		T area() const{
			return width_ * height_;
		}

		T size() const{
			return width_ * height_ * depth_;
		}

		T width_;
		T height_;
		T depth_;
	};

	template <typename T>
	bool operator == (const index3d<T>& lhs, const index3d<T>& rhs){
		return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) && (lhs.depth_ == rhs.depth_);
	}

	template <typename T>
	bool operator != (const index3d<T>& lhs, const index3d<T>& rhs){
		return !(lhs == rhs);
	}

	template <typename Stream, typename T>
	Stream& operator << (Stream& s, const index3d<T>& d){
		s << d.width_ << "x" << d.height_ << "x" << d.depth_;
		return s;
	}
}//namespace bingonet