#include <highvoronoi/detail/locks.hpp>
#include <highvoronoi/detail/read_write_list.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using AddressList = highvoronoi::ReadWriteAddressList<
        std::vector<std::size_t>,
        highvoronoi::detail::ReadWriteLock>;

    AddressList addresses;

    if (addresses.size() != 0) {
        return 1;
    }

    addresses.push_back(10);
    addresses.push_back(20);
    addresses.push_back(30);

    if (addresses.size() != 3) {
        return 2;
    }

    if (addresses[0] != 10 ||
        addresses[1] != 20 ||
        addresses[2] != 30) {
        return 3;
    }

    std::cout << "ReadWriteAddressList test passed.\n";
    return 0;
}
