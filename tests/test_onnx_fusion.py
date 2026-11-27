import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from ab.bot.onnx_fusion import fold_conv_batchnorm


def make_model(*, expose_intermediate=False):
    data = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 2, 3, 3])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 2, 3, 3])
    constants = {
        "weight": [[[[2]]], [[[3]]]],
        "scale": [0.5, 3],
        "offset": [4, -2],
        "mean": [1, 2],
        "variance": [2, 5],
    }
    nodes = [
        helper.make_node("Identity", ["scale"], ["scale_alias"]),
        helper.make_node("Conv", ["image", "weight"], ["intermediate"], group=2),
        helper.make_node(
            "BatchNormalization",
            ["intermediate", "scale_alias", "offset", "mean", "variance"],
            ["result"], epsilon=0.1,
        ),
    ]
    outputs = [result]
    if expose_intermediate:
        outputs.append(helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [1, 2, 3, 3]))
    graph = helper.make_graph(
        nodes, "fusion_test", [data], outputs,
        [numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
         for name, value in constants.items()],
    )
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])


def test_folding_preserves_grouped_conv_outputs_with_alias_and_no_bias():
    model = make_model()
    original = model.SerializeToString()
    assert fold_conv_batchnorm(model) == 1
    assert [node.op_type for node in model.graph.node] == ["Conv"]
    assert fold_conv_batchnorm(model) == 0
    onnx.checker.check_model(model)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    feed = {"image": np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)}
    sessions = [ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
                for data in (original, model.SerializeToString())]
    expected, actual = [session.run(None, feed)[0] for session in sessions]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_folding_skips_convolution_whose_output_is_exposed():
    model = make_model(expose_intermediate=True)
    original = model.SerializeToString()
    assert fold_conv_batchnorm(model) == 0
    assert model.SerializeToString() == original
