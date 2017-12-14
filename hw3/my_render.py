
class my_render():
    def __init__(self):
        from gym.envs.classic_control import rendering
        import numpy as np
        self.viewer = rendering.SimpleImageViewer()
    def render(self, rgb):
        #rgb render
        upscaled=repeat_upsample(rgb,3, 3)
        self.viewer.imshow(upscaled)
        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

    def repeat_upsample(self, rgb_array, k=1, l=1, err=[]):
        # repeat kinda crashes if k/l are zero
        if k <= 0 or l <= 0: 
            if not err: 
                print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
                err.append('logged')
            return rgb_array

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
    def render(self, rgb):
        upscaled=self.repeat_upsample(rgb,3, 3)
        self.viewer.imshow(upscaled)
