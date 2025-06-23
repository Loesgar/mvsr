// A typical double linked list, but enables getting an iterator from an element pointer and holds
// additional space for each element. The list reserves space once and can never grow bigger.
// Otherwise a segfault will occur.

#ifndef MVSR_LIST_HPP
#define MVSR_LIST_HPP

#include <algorithm>
#include <iterator>

template <typename T, typename T2 = char>
class List
{
private:
    struct Element
    {
        T element;
        Element *next = nullptr, *prev = nullptr;
    };
    union Bucket
    {
        Element element;
        T2 _alignment; // aligns the Element to alignment of T2
        size_t _empty; // needed for Element to be at least ptr size
    };

public:
    List(size_t space, size_t reserve = 0) : space(space)
    {
        if (reserve != 0) reserveElements(reserve);
    }
    List(List &&move)
        : first(move.first), last(move.last), space(move.space), size(move.size),
          allocated(move.allocated), freeptr(move.freeptr)
    {
        move.first = move.last = nullptr;
        move.allocated = move.freeptr = nullptr;
        move.size = 0;
    }
    List &operator=(List &&move) & = delete;
    //     {
    //         std::swap(last, move.last);
    //         std::swap(first, move.first);
    //         std::swap(space, move.space);
    //         std::swap(size, move.size);
    //         std::swap(allocated, move.allocated);
    //         std::swap(freeptr, move.freeptr);
    //     }
    List(const List &copy) : space(copy.space)
    {
        clear();
        reserveElements(copy.size);
        for (auto &val : copy)
        {
            auto *to = getExtraData(*append(val));
            auto *from = copy.getExtraData(val);

            std::copy(from, from + space, to);
        }
    }
    List &operator=(const List &copy) & = delete;
    ~List()
    {
        clear();
    }

    void clear()
    {
        for (auto cur = first; cur != nullptr;)
        {
            auto next = cur->next;
            dealloc(cur);
            cur = next;
        }
        ::free(allocated);
        freeptr = allocated = nullptr;
        first = last = nullptr;
        size = 0;
    }

    class Iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using reference = T &;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator &operator++()
        {
            element = element->next;
            return *this;
        }
        Iterator &operator--()
        {
            element = element->prev;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator res(*this);
            ++*this;
            return res;
        }
        Iterator operator--(int)
        {
            Iterator res(*this);
            --*this;
            return res;
        }
        T &operator*() const
        {
            return element->element;
        }
        T *operator->() const
        {
            return &element->element;
        }

        bool operator!=(const Iterator &cmp) const
        {
            return element != cmp.element;
        }
        bool operator==(const Iterator &cmp) const
        {
            return element == cmp.element;
        }

        static Iterator FromElement(T &t)
        {
            return Iterator(&reinterpret_cast<Element &>(t));
        }

    private:
        friend List;
        explicit Iterator(Element *e) : element(e) {}
        Element *element = nullptr;
    };

    T2 *getExtraData(T &e)
    {
        return (T2 *)(((Bucket *)&e) + 1);
    }
    const T2 *getExtraData(const T &e) const
    {
        return (T2 *)(((Bucket *)&e) + 1);
    }

    Iterator begin() const
    {
        return Iterator(first);
    }
    Iterator end() const
    {
        return Iterator(nullptr);
    }
    T &front()
    {
        return first->element;
    }
    T &back()
    {
        return last->element;
    }

    Iterator prepend(T val)
    {
        return insert(begin(), std::move(val));
    }
    Iterator append(T val)
    {
        return insert(end(), std::move(val));
    }
    T popFront()
    {
        return remove(first);
    };
    T popBack()
    {
        return remove(last);
    };

    Iterator insert(const Iterator &pos, T val)
    {
        auto newElement = alloc();
        newElement->element = std::move(val);
        Element *prev, *next;

        if (pos.element == nullptr)
        {
            prev = last;
            next = nullptr;
            last = newElement;
        }
        else
        {
            next = pos.element;
            prev = next->prev;
        }

        if (next != nullptr)
        {
            newElement->next = next;
            next->prev = newElement;
        }
        if (prev != nullptr)
        {
            newElement->prev = prev;
            prev->next = newElement;
        }
        else
        {
            first = newElement;
        }

        ++size;
        return Iterator(newElement);
    }
    T remove(const Iterator &pos)
    {
        auto &oldElement = pos.element;
        if (oldElement->next != nullptr)
        {
            oldElement->next->prev = oldElement->prev;
        }
        else
        {
            last = oldElement->prev;
        }
        if (oldElement->prev != nullptr)
        {
            oldElement->prev->next = oldElement->next;
        }
        else
        {
            first = oldElement->next;
        }
        auto res = std::move(oldElement->element);
        dealloc(oldElement);
        --size;
        return res;
    }
    size_t getSize() const
    {
        return size;
    }
    void reserve(size_t size)
    {
        reserveElements(size);
    }

private:
    size_t getBucketSize() const
    {
        size_t bs = (sizeof(Bucket) + space * sizeof(T2));
        size_t fill = (bs % alignof(Bucket)) == 0 ? 0 : alignof(Bucket) - (bs % alignof(Bucket));
        return bs + fill;
    }
    void reserveElements(size_t reserve)
    {
        if (allocated == nullptr)
        {
            allocated = (Bucket *)::calloc(reserve, getBucketSize());
            freeptr = allocated;
            auto *last = (Bucket *)&((char *)allocated)[(reserve - 1) * getBucketSize()];
            last->_empty =
                (char *)(nullptr) - (char *)(last) - (sizeof(Bucket) + sizeof(T2) * space);
        }
    }
    Element *alloc()
    {
        if (freeptr == nullptr) return nullptr;

        auto next = (Bucket *)((char *)freeptr + freeptr->_empty + getBucketSize());
        Element *res = new (freeptr) Element;
        new (getExtraData(res->element)) T2[space];
        freeptr = next;

        return res;
    }
    void dealloc(Element *ptr)
    {
        for (size_t i = 0; i < space; i++)
        {
            getExtraData(ptr->element)[i].~T2();
        }
        ptr->~Element();

        ((Bucket *)ptr)->_empty = (char *)(freeptr) - (char *)(ptr)-getBucketSize();
        freeptr = (Bucket *)ptr;
    }

    Element *first = nullptr, *last = nullptr;
    Bucket *allocated = nullptr, *freeptr = nullptr;
    size_t size = 0;
    const size_t space;
};

#endif // guard
