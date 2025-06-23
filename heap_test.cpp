#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <iostream>

#include "heap.hpp"

struct RefEntry : Heap<int, RefEntry>::Reference
{
    int a, b, c;
};

Heap<int, RefEntry> h;

void ugly_test(int n)
{
    srand(1337);

    auto max_value = n * 100;
    //h = Heap<int, RefEntry>(n*100)

    std::vector<RefEntry> entries;
    entries.resize(n);
    for (int i = 0; i < entries.size(); i++)
    {
        RefEntry& e  = entries[i];
        e.a = rand() % max_value;
        e.b = rand() % max_value;
        e.c = rand() % max_value;
        h.push(e.b, e);
    }

    int valid_entries = entries.size();

    while (!h.isEmpty())
    {
        if (rand() % 3)
        {
            auto b_comp = [](const RefEntry& a, const RefEntry& b) { return a.b < b.b; };
            auto expected = std::min_element(entries.begin(), entries.end(), b_comp);
            auto pair = h.pop();
            std::cout << "pop " << pair.first << std::endl;
            assert(pair.first == expected->b);
            auto same_b = [pair](const RefEntry& other) { return other.b == pair.first; };
            assert(&pair.second == &*expected || std::count_if(entries.begin(), entries.end(), same_b) > 0);
            expected->a = __INT_MAX__;
            expected->b = __INT_MAX__;
            expected->c = __INT_MAX__;
            valid_entries--;
        }
        else
        {
            size_t actual_index = 0;
            size_t index = valid_entries > 0 ? rand() % valid_entries : 0;
            for (size_t i = 0; i < index || entries[actual_index].b == __INT_MAX__; actual_index++) {
                if (entries[actual_index].b != __INT_MAX__) i++;
            }

            RefEntry& update_entry = entries[actual_index];
            std::cout << "update " << update_entry.b << " => ";
            update_entry.b = rand() % max_value;
            std::cout << update_entry.b << std::endl;
            h.update(update_entry, update_entry.b);
        }
    }
}

int main()
{
    ugly_test(10000);

    return 0;
}
