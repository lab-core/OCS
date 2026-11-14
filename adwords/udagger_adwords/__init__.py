"""Imitation learning for the stochastic AdWords problem.

This package mirrors the physician-to-patient assignment (PPA) study of the
IL4CO project on a second online combinatorial problem: budgeted allocation of
search queries to advertisers (AdWords). It provides a market generator, an
episode simulator with Bayesian scenario sampling, a family of experts that
differ in how they treat uncertainty (myopic, deterministic, full-information,
aggregated deterministic and two-stage stochastic), and a DAgger-based learning
pipeline.
"""
