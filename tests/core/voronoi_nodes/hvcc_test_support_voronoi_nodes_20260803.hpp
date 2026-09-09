#pragma once
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>
namespace hvtest {
using Scalar = double;
using Index = std::size_t;
using IndexVector = std::vector<Index>;
inline constexpr Scalar tolerance = 1e-12;
inline std::size_t performed_checks = 0;
inline std::size_t failed_checks = 0;
inline void check(bool condition, std::string_view description) { ++performed_checks; if (condition) { std::cout << "    [OK]   " << description << '\n'; return; } ++failed_checks; std::cerr << "    [FAIL] " << description << '\n'; }
inline bool close(Scalar left, Scalar right) { return std::abs(left-right) <= tolerance; }
template <typename PointLike> void check_point_2d(const PointLike& point, Scalar expected_x, Scalar expected_y, std::string_view description) { check(close(point[Index{0}],expected_x)&&close(point[Index{1}],expected_y),description); }
template <typename Exception, typename Function> void check_throws(Function&& function, std::string_view description) { bool ok=false; try { std::forward<Function>(function)(); } catch(const Exception&) {ok=true;} catch(...) {} check(ok,description); }
template <typename Function> void run_test(std::string_view name, Function&& function) { std::cout << "\n[TEST] " << name << '\n'; const auto before=failed_checks; try { std::forward<Function>(function)(); } catch(const std::exception& e) { ++failed_checks; std::cerr << "    [FAIL] unexpected exception: " << e.what() << '\n'; } catch(...) { ++failed_checks; std::cerr << "    [FAIL] unexpected non-standard exception\n"; } std::cout << (failed_checks==before ? "[PASS] " : "[FAIL] ") << name << '\n'; }
}
