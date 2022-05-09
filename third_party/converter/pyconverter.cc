/* Copyright (c) Facebook, Inc. and its affiliates. */
#include <cstdio>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "converter.h"

namespace py = pybind11;
using namespace py::literals;

// adapted from pynethack.cc
template <typename T>
T *
checked_conversion(py::handle h, const std::vector<size_t> &shape)
{
    if (h.is_none())
        return nullptr;
    if (!py::isinstance<py::array>(h))
        throw std::invalid_argument("Numpy array required");

    py::array array = py::array::ensure(h);
    // We don't use py::array_t<T> (or <T, 0>) above as that still
    // causes conversions to "larger" types.
    if (!array.dtype().is(py::dtype::of<T>()))
        throw std::invalid_argument("Buffer dtype mismatch.");

    py::buffer_info buf = array.request();

    if (buf.ndim != shape.size()) {
        std::ostringstream ss;
        ss << "Array has wrong number of dimensions (expected "
           << shape.size() << ", got " << buf.ndim << ")";
        throw std::invalid_argument(ss.str());
    }
    if (!std::equal(shape.begin(), shape.end(), buf.shape.begin())) {
        std::ostringstream ss;
        ss << "Array has wrong shape (expected [ ";
        for (auto i : shape)
            ss << i << " ";
        ss << "], got [ ";
        for (auto i : buf.shape)
            ss << i << " ";
        ss << "])";
        throw std::invalid_argument(ss.str());
    }
    if (!(array.flags() & py::array::c_style))
        throw std::invalid_argument("Array isn't C contiguous");

    return static_cast<T *>(buf.ptr);
}
// end of adapted from pynethack.c

class Converter
{
  public:
    Converter(size_t rows, size_t cols, size_t ttyrec_version, size_t term_rows, size_t term_cols
              )
        : rows_(rows), cols_(cols),
          ttyrec_version_(ttyrec_version),
          term_rows_((term_rows != 0) ? term_rows : rows),
          term_cols_((term_cols != 0) ? term_cols : cols)          
    {
        if (term_rows_ < 2 || term_cols_ < 2)
           throw std::invalid_argument("Terminal invalid: term_rows and term_cols must be >1");

        conversion_ = conversion_create(rows_, cols_, term_rows_, term_cols_,
                                        ttyrec_version_);
        if (conversion_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    ~Converter()
    {
        conversion_close(conversion_);
        if (ttyrec_ != nullptr) {
            fclose(ttyrec_);
        }
    }

    void
    load_ttyrec(const std::string filename, size_t gameid, size_t part)
    {
        if (ttyrec_ == nullptr)
            ttyrec_ = fopen(filename.c_str(), "r");
        else
            ttyrec_ = freopen(filename.c_str(), "r", ttyrec_);
        if (ttyrec_ == nullptr) {
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, filename.c_str());
            throw py::error_already_set();
        }

        int status = conversion_load_ttyrec(conversion_, ttyrec_);
        if (status != 0) {
            throw std::runtime_error("File failed to load: '" + filename
                                     + "'");
        }

        gameid_ = gameid;
        part_ = part;
        filename_ = std::move(filename);
    }

    int
    convert(py::object chars, py::object colors, py::object cursors,
            py::object timestamps, py::object inputs, py::object scores)
    {
        int status = 0;

        if (!py::isinstance<py::array>(chars))
            throw std::invalid_argument("Numpy array required");
        py::array array = py::array::ensure(chars);
        if (!array.dtype().is(py::dtype::of<uint8_t>()))
            throw std::invalid_argument("Buffer dtype mismatch.");
        size_t unroll = array.request().shape[0];

        conversion_set_buffers(
            conversion_,
            checked_conversion<uint8_t>(chars, { unroll, rows_, cols_ }),
            unroll * rows_ * cols_,
            checked_conversion<int8_t>(colors, { unroll, rows_, cols_ }),
            unroll * rows_ * cols_,
            checked_conversion<int16_t>(cursors, { unroll, 2 }), unroll * 2,
            checked_conversion<int64_t>(timestamps, { unroll }), unroll,
            checked_conversion<uint8_t>(inputs, { unroll }), unroll,
            checked_conversion<int32_t>(scores, { unroll }), unroll);
        {
            py::gil_scoped_release release;
            status = conversion_convert_frames(conversion_);
        }
        if (status == -1) {
            // TODO : Better error messages
            throw std::runtime_error("Error in file.");
        } else if (status == -2) {
            // ignore: status = -2
            // This occurs when convert is called on a converter that has
            // already been exhausted. This will be common in the last
            // minibatches in a dataset.
        }
        return conversion_->remaining;
    }

    bool
    is_loaded()
    {
        return ttyrec_ != nullptr && filename_ != "";
    }

    const std::string &
    filename()
    {
        return filename_;
    }

    size_t
    gameid()
    {
        return gameid_;
    }

    size_t
    part()
    {
        return part_;
    }

    const size_t rows_ = 0;
    const size_t cols_ = 0;
    const size_t term_rows_ = 0;
    const size_t term_cols_ = 0;
    const size_t ttyrec_version_ = 0;

  private:
    Conversion *conversion_ = nullptr;
    FILE *ttyrec_ = nullptr;

    std::string filename_;
    // These attributes are purely for human readable id of what is loaded
    size_t part_ = 0;
    size_t gameid_ = 0;
};

PYBIND11_MODULE(_pyconverter, m)
{
    m.doc() = "Ttyrec Converter";

    py::class_<Converter>(m, "Converter")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("rows"), py::arg("cols"), py::arg("ttyrec_version"), py::arg("term_rows") = 0,
             py::arg("term_cols") = 0)
        .def("load_ttyrec", &Converter::load_ttyrec, py::arg("filename"),
             py::arg("gameid") = 0, py::arg("part") = 0)
        .def("convert", &Converter::convert, py::arg("chars"),
             py::arg("colors"), py::arg("cursors"), py::arg("timestamps"),
             py::arg("inputs"), py::arg("scores"))
        .def("is_loaded", &Converter::is_loaded)
        .def_readonly("rows", &Converter::rows_)
        .def_readonly("cols", &Converter::cols_)
        .def_readonly("term_rows", &Converter::term_rows_)
        .def_readonly("term_cols", &Converter::term_cols_)
        .def_readonly("ttyrec_version", &Converter::ttyrec_version_)
        .def_property_readonly("filename", &Converter::filename)
        .def_property_readonly("part", &Converter::part)
        .def_property_readonly("gameid", &Converter::gameid);
}