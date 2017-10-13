function mv_print_classification_info(cfg)
% Prints information regarding the classification: prints classifier name
% and cross-validation parameters. Required fields in cfg struct:
% .classifier
% .K
% .CV
% .nRepetitions

fprintf('Performing %s cross-validation (K=%d) with %d repetitions using a %s classifier.\n', ...
    cfg.CV, cfg.K, cfg.repeat, upper(cfg.classifier))