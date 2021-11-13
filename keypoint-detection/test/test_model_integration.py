import unittest

from src.models import KeypointDetector


class TestModelOverfit(unittest.TestCase):
    # TODO:
    def setUp(self) -> None:
        self.model = KeypointDetector()

    def test_overfit(self):
        """
        Verify model can overfit small batch.
        """
        # os.path.dirname(os.path.abspath(__file__))

        # model = KeypointDetector()
        # TEST_DIR = os.path.dirname(os.path.abspath(__file__))

        # module = BoxKeypointsDataModule(
        #     BoxKeypointsDataset(os.path.join(TEST_DIR,"test_dataset/dataset.json"),os.path.join(TEST_DIR,"test_dataset")),
        #     1,
        # )
        # trainer = pl.Trainer(max_epochs=50, log_every_n_steps = 1)
        # trainer.fit(model, module)

        # #batch = next(iter(module.train_dataloader()))
        # print(trainer.logged_metrics)

        # batch = next(iter(module.train_dataloader()))

        # imgs, corner_keypoints, flap_keypoints = batch
        # with torch.no_grad():
        #     predictions = model(imgs)

        # corner_heatmap = predictions[0][0]
        # predicted_corner_keypoints = get_keypoints_from_heatmap(corner_heatmap, 10)

        # print(predicted_corner_keypoints)
        # print(corner_keypoints)
        # self.assertTrue(False)
