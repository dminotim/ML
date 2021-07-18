#pragma  once
#include <vector>
template<typename Scalar>
struct mwOptimizer
{
virtual void Update(const std::vector<Scalar>& grads, std::vector<Scalar>& weights) = 0;
};