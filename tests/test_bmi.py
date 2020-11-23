from brie.brie_bmi import BrieBMI


def test_initialize(tmpdir, datadir):
    with tmpdir.as_cwd():
        brie = BrieBMI()
        brie.initialize(str(datadir / "brie.yaml"))


def test_update(tmpdir, datadir):
    with tmpdir.as_cwd():
        brie = BrieBMI()
        brie.initialize(str(datadir / "brie.yaml"))
        brie.update()
