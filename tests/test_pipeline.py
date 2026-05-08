from main import run


def test_pipeline_end_to_end_writes_loss_curve(small_config, tmp_path):
    small_config["output"]["dir"] = str(tmp_path)

    run(small_config)

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
