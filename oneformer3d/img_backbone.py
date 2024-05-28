from ultralytics import YOLO
from PIL import Image
import torchvision
import pdb
from ultralytics.yolo.utils import is_git_dir
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.v8.segment import SegmentationPredictor
from ultralytics.yolo.cfg import get_cfg
import sys
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

class MyYOLO(YOLO):
    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = MyPredictor(overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
            if 'project' in overrides or 'name' in overrides:
                self.predictor.save_dir = self.predictor.get_save_dir()
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)


class MyPredictor(SegmentationPredictor):
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment)
        return preds

@MODELS.register_module()
class FastSAM_Backbone(BaseModule):
    def __init__(self):
        super(FastSAM_Backbone, self).__init__()
        self.yolo = MyYOLO('/home/ubuntu/xxw/OS3D/FastSAM/FastSAM-x.pt')
        self.yolo.model.model = self.yolo.model.model[:15]

    def init_weights(self):
        for param in self.yolo.model.parameters():
            param.requires_grad = False
        self.yolo.model.eval()

    def forward(self, x):
        x = self.yolo.predict(x, device='cuda', retina_masks=True, imgsz=640)
        return x


# yolo = MyYOLO('weights/FastSAM-x.pt')
# yolo.model.model = yolo.model.model[:15]
# embedding = yolo.predict('images/0.jpg', device='cuda', retina_masks=True, imgsz=640)[0]
# print(embedding.shape)

#embedding = yolo.model(tensor_img)
#print(embedding.shape)
