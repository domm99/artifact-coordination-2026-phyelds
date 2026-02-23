import imageio
from PIL import Image
from phyelds.simulator import Simulator, Monitor
from phyelds.simulator.effects import RenderConfig


class VMASRenderMonitor(Monitor):

    def __init__(self, simulator: Simulator, config: RenderConfig):
        super().__init__(simulator)
        self.config = config
        self.simulator = simulator
        self.frames = []
        self.tick = 0

    def update(self):
        if self.tick % 20 == 0:
            f = self.simulator.environment.vmas_environment.render(mode="rgb_array")
            self.frames.append(f)
        self.tick += 1

    def on_finish(self):
        imageio.mimsave(f"{self.config.save_as}.gif", self.frames, fps=20)

        for index, frame in enumerate(self.frames):
            img = Image.fromarray(frame)
            name = f'output/frame-{index:04d}.pdf'
            img.save(name, resolution=500)