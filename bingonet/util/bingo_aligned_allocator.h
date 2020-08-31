//edited by bingo
//todo

#pragma once
#include <stdlib.h>
#include "nn_error.h"

namespace bingonet{
	template <typename T, std::size_t alignment>
	class aligned_allocator{
	public:
		typedef T value_type;
		typedef T* pointer;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef const T* const_pointer;

		template <typename U>
		struct rebind{
			typedef aligned_allocator<U, alignment> other;
		};

		aligned_allocator() {}

		aligned_allocator(const aligned_allocator& ) {}

		template <typename U>
		aligned_allocator(const aligned_allocator<U, alignment>&) {}

		~aligned_allocator() {}

		pointer address(reference x) const {
			return std::addressof(x);
		}

		const_pointer address(const_reference x) const {
			return std::addressof(x);
		}

		pointer allocate(size_type n, const void* = nullptr){
			void* p = aligned_alloc(alignment, sizeof(T) * n);
			if(!p && n > 0)
				throw nn_error("failed to allocate");
			return static_cast<pointer>(p);
		}

		void deallocate(pointer p, size_type n){
			aligned_free(p);
		}

		size_type max_size() const {
			return ~static_cast<std::size_t>(0) / sizeof(T);
		}

		template<class U, class V>
		void construct(U* p, const V& x){
			void* ptr = p;
			::new(ptr) U(x);
		}

		template<class U, class... Args>
		void construct(U* p, Args&&... args){
			void* ptr = p;
			::new(ptr) U(std::forward<Args>(args)...);
		} 

		template<class U>
		void construct(U* p){
			void* ptr = p;
			::new(ptr) U();
		}

		template<class U>
		void destroy(U* p){
			p->~U();
		}
	private:

		void* aligned_alloc(size_type align, size_type size) const{
			void* ptr;
			if(::posix_memalign(&ptr, align, size) != 0){
				ptr = 0;
			}
			return ptr;
		}

		void* aligned_free(pointer p){
			::free(p);
		}
	};

	template<typename T1, typename T2, std::size_t alignment>
	inline bool operator==(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&){
		return true;
	}

	template<typename T1, typename T2, std::size_t alignment>
	inline bool operator!=(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&){
		return false;
	}
}//namespace bingonet