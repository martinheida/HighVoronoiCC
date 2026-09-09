/*#include <iostream>
#include <vector>
#include <highvoronoi/core/point.hpp>

int main()
{
    using P = highvoronoi::StaticPoint<double, 3>;

    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(P)      = " << sizeof(P) << '\n';
    std::cout << "alignof(P)     = " << alignof(P) << '\n';

    std::vector<P> points(3);

    std::cout << "points.data()              = "
              << static_cast<void*>(points.data()) << '\n';

    std::cout << "points[0].data()           = "
              << static_cast<void*>(points[0].data()) << '\n';

    std::cout << "points[1].data()           = "
              << static_cast<void*>(points[1].data()) << '\n';

    std::cout << "distance in bytes          = "
              << reinterpret_cast<char*>(points[1].data())
               - reinterpret_cast<char*>(points[0].data())
              << '\n';

    static_assert(
        sizeof(P) == 3 * sizeof(double),
        "StaticPoint<double,3> contains padding!"
    );
}
*/
#include <iostream>
#include <Eigen/Core>

#include <highvoronoi/core/point.hpp>

int main()
{
    double data[6] = {1.0, 2.0, 3.0,1.0, 2.0, 3.0};

    highvoronoi::StaticPointView<double, 3> view(data+1);

    std::cout << "Before:\n";
    std::cout << view << "\n\n";

    // Modify the original memory.
    data[1] = 42.0;

    std::cout << "After changing data[1]:\n";
    std::cout << view << '\n';
}
