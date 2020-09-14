from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()



# -----------------------------------------------------------------------------
# META learning
# -----------------------------------------------------------------------------
_C.META = CN()
_C.META.BOTTLENECK = CN()
_C.META.BOTTLENECK.DO_IT = True
_C.META.BOTTLENECK.REDUCTION_DIM = 1024
_C.META.BOTTLENECK.NORM = False
_C.META.GRL = CN()
_C.META.GRL.DO_IT = False
_C.META.GRL.LOCATION = "after" # 'before' / 'after'
_C.META.GRL.GAMMA = 10
_C.META.GRL.WEIGHT = 1.0
_C.META.GRL.LR_FACTOR = 1.0
_C.META.DATA = CN()
_C.META.DATA.NAMES = "VeRi_keypoint_each_4"
_C.META.DATA.NUM_DOMAINS = 4
_C.META.DATA.RELABEL = False
_C.META.DATA.DROP_LAST = True

_C.META.REGULARIZER = CN()
_C.META.REGULARIZER.DO_IT = True
_C.META.REGULARIZER.LR_FACTOR = 10.0 # including "reg", and check weight decay
_C.META.MODEL = CN()
_C.META.MODEL.LR_FACTOR = 1.0 # including "meta"
# _C.META.MODEL.SPLIT_LAYER = ("conv1","conv2","conv3","conv4","aver_pool","bottleneck","last_fc")
_C.META.MODEL.SPLIT_LAYER = ("bottleneck",) # "bottleneck","classifier","bnneck","pooling",
_C.META.MODEL.SAME_NUM_CE = False # operate when SPLIT classifier is True => True -> In META-test, CE does not work
# _C.META.MODEL.LEARNABLE_LAYER = ("",) # "bnneck","pooling", do not duplicate with SPLIT_LAYER
# _C.META.MODEL.POOLING_REG_CONSTRAINT = 'free' # 'free', 'lower_bound', 'both_bound'
# _C.META.MODEL.POOLING_WEIGHT_CONSTRAINT = 'free' # 'spatial', 'channel', 'sc'
# _C.META.MODEL.BN_REG_CONSTRAINT = 'free' # 'free', 'lower_bound', 'both_bound'

_C.META.SOLVER = CN()
_C.META.SOLVER.INIT = CN()

_C.META.SOLVER.INIT.MAX_ITER = 0
_C.META.SOLVER.INIT.SCHED = 'WarmupMultiStepLR'
_C.META.SOLVER.INIT.GAMMA = 0.1
_C.META.SOLVER.INIT.STEPS = [20, 40]
_C.META.SOLVER.INIT.WARMUP_ITERS = 10

_C.META.SOLVER.INIT.IMS_PER_BATCH = 64
_C.META.SOLVER.INIT.NUM_INSTANCE = 4
_C.META.SOLVER.INIT.INNER_LOOP = 1
_C.META.SOLVER.INIT.OUTER_LOOP = 5

_C.META.SOLVER.MTRAIN = CN()
_C.META.SOLVER.MTRAIN.IMS_PER_BATCH = 64
_C.META.SOLVER.MTRAIN.NUM_INSTANCE = 4
_C.META.SOLVER.MTRAIN.INNER_LOOP = 1 # not working
_C.META.SOLVER.MTRAIN.INNER_LOOP_TYPE = 'same' # same: update w/o step, diff: step and update
_C.META.SOLVER.MTRAIN.NUM_DOMAIN = 1

_C.META.SOLVER.MTEST = CN()
_C.META.SOLVER.MTEST.IMS_PER_BATCH = 64
_C.META.SOLVER.MTEST.NUM_INSTANCE = 4
_C.META.SOLVER.MTEST.ONLY_ONE_DOMAIN = False # True: only batch IMG_PER_BATCH, False: batch*=(num_view-num_domain)

_C.META.SOLVER.FINAL = CN()

_C.META.SOLVER.FINAL.MAX_ITER = 60 # based on meta-test set
_C.META.SOLVER.FINAL.SCHED = 'WarmupMultiStepLR'
# _C.META.SOLVER.FINAL.SCHED = 'constant'
_C.META.SOLVER.FINAL.GAMMA = 0.1
_C.META.SOLVER.FINAL.STEPS = [20, 40]
_C.META.SOLVER.FINAL.WARMUP_ITERS = 10

_C.META.SOLVER.FINAL.INITIALIZE_CONV_STATE_META = 'pretrained' # pretrained / scratch
_C.META.SOLVER.FINAL.INITIALIZE_CONV_STATE_ORI = 'pretrained' # pretrained / scratch
_C.META.SOLVER.FINAL.INITIALIZE_CONV = True # False: load meta conv layers / true: choose state ORI
_C.META.SOLVER.FINAL.INITIALIZE_FC = True # False: load meta fc-layers / true: weight initialization (remain ORI)
_C.META.SOLVER.FINAL.DETAIL_MODE = False
_C.META.SOLVER.FINAL.WEIGHT_LOADER = None # if exists, proceed ori-learning
_C.META.SOLVER.FINAL.WEIGHT_FOLDER = None # if exists, proceed ori-learning
_C.META.SOLVER.FINAL.SAVE_META_PARAM = True
_C.META.SOLVER.TRAIN_ORIGINAL = True
_C.META.SOLVER.SYNC = True
_C.META.SOLVER.WRITE_PERIOD = 20
_C.META.SOLVER.WRITE_PERIOD_PARAM = 5

_C.META.SOLVER.BIN_GATE = CN()
_C.META.SOLVER.BIN_GATE.LR_FACTOR = 100.0

_C.META.LOSS = CN()
_C.META.LOSS.COMBINED = False
_C.META.LOSS.META_REG_WEIGHT = 0.001
_C.META.LOSS.ORI_REG_WEIGHT = 0.001
_C.META.LOSS.INIT_NAME = ("CrossEntropyLoss", "TripletLoss",)
_C.META.LOSS.MTRAIN_NAME = ("CrossEntropyLoss", "TripletLoss", "Reg_bottleneck", "Reg_classifier",)
_C.META.LOSS.MTEST_NAME = ("CrossEntropyLoss", "TripletLoss",)
# _C.META.LOSS.INIT_NAME = ("TripletLoss",)
# _C.META.LOSS.MTRAIN_NAME = ("TripletLoss", "Reg",)
# _C.META.LOSS.MTEST_NAME = ("TripletLoss", )







# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = 'Baseline'
_C.MODEL.FREEZE_LAYERS = ['']



# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = 50
# RegNet volume
_C.MODEL.BACKBONE.VOLUME = "800y"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# Mini-batch split of Ghost BN
_C.MODEL.BACKBONE.NORM_SPLIT = 1
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = True
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''



# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "BNneckHead"

# Normalization method for the convolution layers.
# _C.MODEL.HEADS.NORM = "BN"
_C.MODEL.HEADS.NORM = "BN"
_C.MODEL.HEADS.BT_NORM = "BN"
# Mini-batch split of Ghost BN
_C.MODEL.HEADS.NORM_SPLIT = 1
# Number of identity
_C.MODEL.HEADS.NORM_INIT_KAIMING = True
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 512
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss", "TripletLoss", "Reg_bottleneck", "Reg_classifier",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.1
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.ALPHA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]



# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.8
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1
# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = True
_C.INPUT.REA.PROB = 0.5
# _C.INPUT.REA.MEAN = [0.596*255, 0.558*255, 0.497*255]  # [0.485*255, 0.456*255, 0.406*255]
_C.INPUT.REA.MEAN = [123.675, 116.28, 103.53]
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = True
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.AMP = True

_C.SOLVER.OPT = "SGD"

_C.SOLVER.MAX_ITER = 100

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 2.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9
# _C.SOLVER.MOMENTUM = 0

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# Multi-step learning rate options
_C.SOLVER.SCHED = "WarmupMultiStepLR"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [40, 70]

# Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 0
_C.SOLVER.ETA_MIN_LR = 7.7e-5 # 3e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.FREEZE_ITERS = 0

# SWA options
_C.SOLVER.SWA = CN()
_C.SOLVER.SWA.ENABLED = False
_C.SOLVER.SWA.ITER = 10
_C.SOLVER.SWA.PERIOD = 2
_C.SOLVER.SWA.LR_FACTOR = 10.
_C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
_C.SOLVER.SWA.LR_SCHED = False

_C.SOLVER.CHECKPOINT_PERIOD = 20

_C.SOLVER.WRITE_PERIOD = 10
_C.SOLVER.WRITE_PERIOD_PARAM = 5

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 10

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.METRIC = "cosine"
_C.TEST.REPORT_ALL = True

# Average query expansion
_C.TEST.AQE = CN()
_C.TEST.AQE.ENABLED = False
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CN()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True

