class LFModelSimple:
    def __init__(self, params):
        self.params = params

    def compute_dx(self, data):
        p = self.params

        u1 = data['u1']
        u2 = data['u2']
        v = data['v']
        a = data['a']
        w = data['w']
        v1 = v + w*p[0]
        v2 = v - w*p[0]

        am1 = p[1]*u1 + p[2]*v1
        gt0 = am1 > 0
        am1[gt0] = (am1[gt0] - p[4]).max(0)
        am1[~gt0] = (am1[~gt0] + p[4]).min(0)

        am2 = p[1]*u2 + p[2]*v2
        gt0 = am2 > 0
        am2[gt0] = (am2[gt0] - p[4]).max(0)
        am2[~gt0] = (am2[~gt0] + p[4]).min(0)

        vp = am1 + am2
        wp = p[3] * (am1 - am2)
        return {
            'v': vp,
            'w': wp,
            'a': w,
            'x': v * a.cos(),
            'y': v * a.sin(),
        }
