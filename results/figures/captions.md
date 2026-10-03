All panels produced under pre-registration 864ec5131b28 (full hash 864ec5131b281d6cf3d48c481c897acd803d2e93). n = 30 tasks, n_eff = 10.0 at an assumed intra-receptor-class correlation of 0.3.

**fig1** Target assortativity against the pointwise-minus-pairwise gap, one point per target, coloured by receptor class. The interval is a cluster bootstrap over receptor classes, not over rows. The panel licenses a statement about association across tasks; it does not license a causal claim, and it does not by itself show the statistic adds anything to roughness, which is the separate incremental quantity in the report.

**fig2** The gap split into what test-time label access buys and what the learned pairwise function buys, per target. Heights are error differences in the label's units. They are **not** shares of the gap: RMSE is nonlinear, so a percentage would be meaningless.

**fig3** Hodge subspace dimensions per target on the pre-registered kNN graph. Hatched bars mark targets whose curl rank was estimated by a randomised range finder rather than computed exactly, which happens above the stated edge count. An estimated bar does not license the precision an exact one does.

**fig4** Curl fraction of the trained edge flow under a bias-free linear readout and under an MLP readout, both on the frozen fingerprint encoder, log scale with the machine-zero line drawn. The linear result is a theorem being checked, not a measurement; the MLP magnitudes are measurements and are specific to this encoder and these graphs.