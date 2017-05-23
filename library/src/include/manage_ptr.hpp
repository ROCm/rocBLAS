#ifndef GUARD_ROCBLAS_MANAGE_PTR_HPP
#define GUARD_ROCBLAS_MANAGE_PTR_HPP

#include <memory>
#include <type_traits>

namespace rocblas
{
	template<class F, F f>
	struct manage_deleter
	{
		template<class T>
		void operator()(T* x) const
		{
			if (x != nullptr) { f(x); }
		}
	};

	template<class T, class F, F f>
	using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

} // namespace rocblas

#define ROCBLAS_MANAGE_PTR(T, F) rocblas::manage_ptr<typename std::remove_pointer<T>::type, decltype(&F), &F>

#endif