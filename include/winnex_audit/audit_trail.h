#ifndef WINNEX_AUDIT_TRAIL_H
#define WINNEX_AUDIT_TRAIL_H

#include <string>
#include <vector>
#include "engine.h"

namespace winnex {

/// JSON formatter for audit records
class AuditJsonFormatter {
public:
    /// Format a single audit record as JSON line
    static std::string record_to_json(const AuditRecord& rec);

    /// Format full audit result as JSON object
    static std::string result_to_json(
        const AuditResult& result,
        const std::string& query_id = "unknown");

    /// Format regulatory compliance report
    static std::string compliance_report(
        const std::vector<AuditResult>& results,
        const std::string& regulation = "EU_AI_ACT");
};

/// Compliance checker against regulatory requirements
class ComplianceChecker {
public:
    struct Report {
        bool bound_guarantee_verified = false;
        int64_t total_violations = 0;
        int64_t total_pairs_checked = 0;
        double violation_rate = 0.0;
        bool deterministic = true;
        bool audit_trail_available = true;
        std::string eu_ai_act_status;
        std::string lgpd_status;
        std::string hipaa_status;
    };

    /// Run compliance check across multiple queries
    static Report check_compliance(
        const std::vector<AuditResult>& results);

    /// Generate compliance certificate summary
    static std::string compliance_certificate(const Report& report);
};

} // namespace winnex
#endif // WINNEX_AUDIT_TRAIL_H
