How to add new set of features:
(1) Go to directory ../utilities/
(2) Create a file named initialize<feature_name_abbreviation>LearnObsAvoidFeatParam.m (or copy and rename from existing ones).
(3) Inside this initialize<feature_name_abbreviation>LearnObsAvoidFeatParam.m file define all parameters needed to compute the feature set (again see example of the existing ones).
(4) Edit a file named getLOA_FeatureDimensionPerPoint.m, to update some loa_feat_param parameter values. Define a new loa_feat_method (that hasn't been used/defined there before) to become the ID of the new feature set (again see example of the existing ones in this file).
(5) Edit a file named initializeAllInvolvedLOAparams.m: define the parameter settings (grids, etc.) and call the feature set initialization function initialize<feature_name_abbreviation>LearnObsAvoidFeatParam() which was created in step (2), with its defined feature parameter settings as its arguments.
(6) Create/copy&edit a file named compute<feature_name_abbreviation>ObstAvoidCtFeatPerPoint.m. This is where the actual feature computation happens.
(7) Edit a file named computeObsAvoidCtFeatPerPoint.m. Use the same feature set ID as defined in step (4) for the selection IF-ELSE statement in this file (around line 39 or so).
(8) Finally in ../learn_obs_avoid_vicon_data/learn_obs_avoid_vicon_data.m, add the description of the new feature set ID as comments in around line 42 or so, and select the new feature set by including its ID in loa_feat_methods definition (somewhere between line 50~70).
