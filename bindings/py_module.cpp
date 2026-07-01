#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "winnex_audit/engine.h"

namespace py = pybind11;

PYBIND11_MODULE(winnex_audit_py, m) {
    m.doc() = "Winnex Audit Module - Mathematical proof layer for vector search";

    py::class_<winnex::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("input_dim", &winnex::Config::input_dim)
        .def_readwrite("stage1_dim", &winnex::Config::stage1_dim)
        .def_readwrite("stage2_dim", &winnex::Config::stage2_dim)
        .def_readwrite("final_k", &winnex::Config::final_k)
        .def_readwrite("stage2_topk", &winnex::Config::stage2_topk);

    py::class_<winnex::AuditRecord>(m, "AuditRecord")
        .def_readonly("doc_id", &winnex::AuditRecord::doc_id)
        .def_readonly("true_cosine", &winnex::AuditRecord::true_cosine)
        .def_readonly("projected_cosine", &winnex::AuditRecord::projected_cosine)
        .def_readonly("residual_norm", &winnex::AuditRecord::residual_norm)
        .def_readonly("upper_bound", &winnex::AuditRecord::upper_bound)
        .def_readonly("threshold", &winnex::AuditRecord::threshold)
        .def_readonly("excluded", &winnex::AuditRecord::excluded)
        .def_readonly("stage", &winnex::AuditRecord::stage);

    py::class_<winnex::AuditResult>(m, "AuditResult")
        .def_readonly("indices", &winnex::AuditResult::indices)
        .def_readonly("scores", &winnex::AuditResult::scores)
        .def_readonly("audit", &winnex::AuditResult::audit)
        .def_readonly("latency_ms", &winnex::AuditResult::latency_ms);

    py::class_<winnex::AuditEngine>(m, "AuditEngine")
        .def(py::init<const winnex::Config&>(), py::arg("config") = winnex::Config())
        .def("build", [](winnex::AuditEngine& self, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be 2-dimensional");
            self.build(static_cast<const float*>(buf.ptr), buf.shape[0]);
        })
        .def("search", [](winnex::AuditEngine& self, py::array_t<float> query, int64_t k) {
            py::buffer_info buf = query.request();
            return self.search(static_cast<const float*>(buf.ptr), k);
        }, py::arg("query"), py::arg("k") = 10)
        .def("check_bounds", [](winnex::AuditEngine& self, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            return self.check_bounds(static_cast<const float*>(buf.ptr));
        })
        .def("audit_json", [](winnex::AuditEngine& self, py::array_t<float> query, int64_t k) {
            py::buffer_info buf = query.request();
            return self.audit_json(static_cast<const float*>(buf.ptr), k);
        }, py::arg("query"), py::arg("k") = 10)
        .def_property_readonly("size", &winnex::AuditEngine::size)
        .def_property_readonly("build_time", &winnex::AuditEngine::build_time);

    m.def("verify_with_faiss", [](py::array_t<float> vectors, py::array_t<float> queries, int64_t k) {
        // Integration example: use FAISS for search, Winnex for audit
        py::gil_scoped_release release;
        auto v_buf = vectors.request();
        auto q_buf = queries.request();

        winnex::Config cfg;
        cfg.input_dim = v_buf.shape[1];

        winnex::AuditEngine engine(cfg);
        engine.build(static_cast<const float*>(v_buf.ptr), v_buf.shape[0]);

        // For each query: FAISS search + Winnex audit
        py::list results;
        int64_t nq = q_buf.shape[0];
        for (int64_t qi = 0; qi < nq; ++qi) {
            const float* q = static_cast<const float*>(q_buf.ptr) + qi * cfg.input_dim;
            auto audit_result = engine.search(q, k);
            auto [v64, v128] = engine.check_bounds(q);
            results.append(audit_result);
        }
        return results;
    }, py::arg("vectors"), py::arg("queries"), py::arg("k") = 10);
}
