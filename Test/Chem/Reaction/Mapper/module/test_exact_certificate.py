from synkit.Chem.Reaction.Mapper.exact.certificate import Certificate


def test_certificate_reports_optimal_when_bounds_meet():
    certificate = Certificate(upper_bound=2.0, lower_bound=2.0, method="test")

    assert certificate.proven_optimal
    assert certificate.gap == 0.0
