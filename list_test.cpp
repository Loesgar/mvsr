#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <iterator>
#include <vector>

#include "list.hpp"

void compare(List<int> &l, std::vector<int> &v)
{
    size_t i = 0;
    for (auto val : l)
    {
        if (val != v[i++])
        {
            std::cerr << "Missmatch" << std::endl;
            throw;
        }
    }
}

void ugly_test(int n)
{
    srand(1337);
    auto max_value = n * 100;

    std::vector<int> v;
    List<int> l;
    for (int i = 0; i < n; i++)
    {
        int val = rand();
        v.push_back(val);
        l.append(val);
    }

    compare(l, v);

    while (!v.empty())
    {
        if (rand() % 3 == 0)
        {
            int pos = (rand() % (v.size() + 1));
            int val = rand();
            v.insert(v.begin() + pos, val);
            auto it = l.begin();
            for (auto i = 0; i < pos; i++) { ++it; }
            l.insert(it, val);
        }
        else
        {
            int pos = (rand() % (v.size()));
            int val = rand();
            v.erase(v.begin() + pos);
            auto it = l.begin();
            for (auto i = 0; i < pos; i++) { ++it; }
            l.remove(it);
        }
        compare(l, v);
    }
}

int main()
{
    ugly_test(100000);
    return 0;
}
