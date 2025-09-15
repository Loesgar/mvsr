// A typical binary heap, but using reference tracking (if the Val type derives
// from Heap::Entry), in order to allow updates of values in the heap.

#ifndef MVSR_HEAP_HPP
#define MVSR_HEAP_HPP

#include <type_traits>
#include <vector>

template <typename Key, typename Val>
class Heap
{
    struct Entry
    {
        Entry() = default;
        ~Entry() = default;
        Entry(const Entry &) = delete;
        Entry &operator=(const Entry &) & = delete;
        Entry(Key k, Val &v) : k(std::move(k)), v(&v)
        {
            if constexpr (std::is_convertible_v<Val &, Reference &>)
            {
                static_cast<const Reference &>(v).e = this;
            }
        }
        Entry(Entry &&e) : k(std::move(e.k)), v(std::move(e.v))
        {
            if constexpr (std::is_convertible_v<Val &, Reference &>)
            {
                static_cast<const Reference &>(*v).e = this;
            }
        }
        Entry &operator=(Entry &&e)
        {
            k = std::move(e.k);
            v = std::move(e.v);
            if constexpr (std::is_convertible_v<Val &, Reference &>)
            {
                static_cast<const Reference &>(*v).e = this;
            }
            return *this;
        }
        Key k;
        Val *v;
    };

public:
    Heap() = default;
    Heap(Heap &&) = default;
    Heap<Key, Val> &operator=(Heap<Key, Val> &&) = default;
    ~Heap() = default;

    Heap(const Heap &) = delete;
    Heap<Key, Val> &operator=(const Heap<Key, Val> &) = delete;

    Heap(size_t size) : data()
    {
        data.reserve(size + 1);
    }

    template <typename R1, typename R2>
    Heap copyByOrder(const R1 &from, const R2 &to) const
    {
        Heap res;
        res.data.resize(data.size());

        auto itFrom = from.begin();
        auto itTo = to.begin();
        for (size_t i = 0; i < data.size(); i++, ++itFrom, ++itTo)
        {
            // if (itFrom == from.end() || itTo == to.end()) throw;
            auto srcEntry = static_cast<const Reference &>(*itFrom).e;
            // if (srcEntry == nullptr) throw;
            res.data[srcEntry - data.data()] = Entry(srcEntry->k, *itTo);
        }
        return res;
    }

    class Reference
    {
    private:
        mutable Entry *e = nullptr;
        friend Heap<Key, Val>;
        friend Heap<Key, Val>::Entry;
    };

    bool isEmpty()
    {
        return data.empty();
    }
    void clear()
    {
        data.clear();
    }
    std::pair<Key, Val &> peek()
    {
        return {data.front().k, *data.front().v};
    }
    std::pair<Key, Val &> pop()
    {
        std::swap(data.front(), data.back());
        auto res = std::move(data.back());
        data.resize(data.size() - 1);
        if (!isEmpty())
        {
            updateDown(0);
        }

        if constexpr (std::is_convertible_v<Val &, Reference &>)
        {
            static_cast<const Reference &>(*res.v).e = nullptr;
        }

        return {res.k, *res.v};
    }
    void update(Val &elem, Key newkey)
    {
        if (auto &entry = static_cast<Reference &>(elem).e; entry == nullptr)
        {
            push(newkey, elem);
        }
        else
        {
            bool up = (newkey < entry->k);
            entry->k = newkey;
            size_t idx = entry - data.data();
            up ? updateUp(idx) : updateDown(idx);
        }
    }
    std::pair<Key, Val &> remove(Val &elem)
    {
        auto &entry = static_cast<Reference &>(elem).e;
        if (auto &entry = static_cast<Reference &>(elem).e; entry != nullptr)
        {
            size_t idx = entry - data.data();
            std::swap(data[idx], data.back());
            auto res = std::move(data.back());
            data.resize(data.size() - 1);
            if (!isEmpty())
            {
                updateDown(idx);
            }
            if constexpr (std::is_convertible_v<Val &, Reference &>)
            {
                static_cast<const Reference &>(*res.v).e = nullptr;
            }
            return {res.k, *res.v};
        }
        return {Key(), elem};
    }
    void push(Key k, Val &v)
    {
        pushBack(k, v);
        updateUp(data.size() - 1);
    }

    void pushProvisionally(Key k, Val &v)
    {
        pushBack(k, v);
    }
    void heapify()
    {
        for (size_t i = data.size(); i-- != 0; updateDown(i));
    }

    void reserve(size_t n)
    {
        data.reserve(n);
    }
    size_t getSize() const
    {
        return data.size();
    }

private:
    void updateUp(size_t idx)
    {
        ++idx;
        auto d = (&data[0]) - 1;
        auto elem = std::move(d[idx]);
        for (; idx > 1; idx /= 2)
        {
            if (d[idx / 2].k < elem.k) break;
            d[idx] = std::move(d[idx / 2]);
        }
        d[idx] = std::move(elem);
    }
    void updateDown(size_t idx)
    {
        ++idx;
        auto d = (&data[0]) - 1;
        auto elem = std::move(d[idx]);
        for (auto child = idx * 2; child <= data.size(); idx = child, child *= 2)
        {
            if (child + 1 <= data.size() && d[child + 1].k < d[child].k) ++child;
            if (!(d[child].k < elem.k)) break;
            d[idx] = std::move(d[child]);
        }
        d[idx] = std::move(elem);
    }
    void pushBack(Key k, Val &v)
    {
        data.push_back(Entry{k, v});
    }

    std::vector<Entry> data;
};

#endif // guard
