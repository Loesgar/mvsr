// A typical double linked list, but enabling to get an iterator from an element pointer

#ifndef LIST_HPP
#define LIST_HPP

#include <algorithm>
#include <iterator>

// todo: optimization: Pool allocator (initial size is always known in our usecase and only shrinks)

template <typename T>
class List
{
private:
    struct Element
    {
        T element;
        Element *next = nullptr, *prev = nullptr;
    };
    Element *_first = nullptr, *_last = nullptr;

public:
    List() = default;
    List(List &&move) : _first(move._first), _last(move._last)
    {
        move._first = move._last = nullptr;
    }
    List &operator=(List &&move) &
    {
        std::swap(_last, move._last);
        std::swap(_first, move._first);
    }
    List(const List &copy) { *this = copy; }
    List &operator=(const List &copy) &
    {
        clear();
        for (auto &val : copy)
        {
            append(val);
        }
        return *this;
    }
    ~List()
    {
        clear();
    }

    void clear()
    {
        for (auto cur = _first; cur != nullptr;)
        {
            auto next = cur->next;
            delete cur;
            cur = next;
        }
        _first = _last = nullptr;
    }

    class Iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator &operator++() { element = element->next; return *this; }
        Iterator &operator--() { element = element->prev; return *this; }
        Iterator operator++(int) { Iterator res(*this); ++*this; return res; }
        Iterator operator--(int) { Iterator res(*this); --*this; return res; }

        T &operator*() const { return element->element; }
        T *operator->() const { return &element->element; }

        bool operator!=(const Iterator &cmp) const { return element != cmp.element; }
        bool operator==(const Iterator &cmp) const { return element == cmp.element; }
        operator bool() const { return element != nullptr; }

        static Iterator FromElement(T &t) { return Iterator(&reinterpret_cast<Element &>(t)); }
    protected:
        friend List;
        explicit Iterator(Element *e) : element(e) {}
        Element *element = nullptr;
    };

    class RIterator : protected Iterator
    {
    public:
        using typename Iterator::difference_type;
        using typename Iterator::value_type;
        using typename Iterator::iterator_category;

        T &operator*() const { return Iterator::element->element; }
        T *operator->() const { return &Iterator::element->element; }

        bool operator!=(const RIterator &cmp) const { return Iterator::element != cmp.element; }
        bool operator==(const RIterator &cmp) const { return Iterator::element == cmp.element; }
        operator bool() const { return Iterator::element != nullptr; }

        RIterator &operator++() { Iterator::operator--(); return *this; }
        RIterator &operator--() { Iterator::operator++(); return *this; }
        RIterator operator++(int) { RIterator res(*this); Iterator::operator--(0); return res; }
        RIterator operator--(int) { RIterator res(*this); Iterator::operator++(0); return res; }

        static RIterator FromElement(T &t) { return RIterator(&reinterpret_cast<Element &>(t)); }
    protected:
        friend List;
        explicit RIterator(Element *e) : Iterator(e) {}
    };

    Iterator begin() const { return Iterator(_first); }
    Iterator end() const { return Iterator(nullptr); }
    RIterator rbegin() const { return RIterator(_last); }
    RIterator rend() const { return RIterator(nullptr); }
    T &front() { return _first->element; }
    T &back() { return _last->element; }

    Iterator prepend(T val) { return insert(begin(), std::move(val)); }
    Iterator append(T val) { return insert(end(), std::move(val)); }
    T popFront() { return remove(_first); };
    T popBack() { return remove(_last); };

    Iterator insert(const Iterator &pos, T val)
    {
        auto newElement = new Element(std::move(val));
        Element *prev, *next;

        if (pos.element == nullptr)
        {
            prev = _last;
            next = nullptr;
            _last = newElement;
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
            _first = newElement;
        }

        return Iterator(newElement);
    }
    T remove(const Iterator &pos)
    {
        auto oldElement = pos.element;
        if (oldElement->next != nullptr) { oldElement->next->prev = oldElement->prev; }
        else { _last = oldElement->prev; }
        if (oldElement->prev != nullptr) { oldElement->prev->next = oldElement->next; }
        else { _first = oldElement->next; }
        auto res = std::move(oldElement->element);
        delete oldElement;
        return res;
    }
};

#endif // guard
