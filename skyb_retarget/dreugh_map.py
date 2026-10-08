"""Land Dreugh: Skyblivion rig -> Oblivion skeleton correspondence (hand-authored).

SKIN_MAP sends every Skyblivion skin bone to the Oblivion bone that takes its
weights.  LANDMARKS places Oblivion bones onto the Skyblivion model, as
('boundary', a, b) -- centroid of the vertices weighted to both a and b, the
joint the artist painted -- or ('tip', bones) -- the lowest vertices of those
bones (a foot tip) -- or ('reach', bone, frac) -- `frac` of the way from that
bone's joint to its farthest vertices.  SWING bones also turn so their segment
points at the named child's new position.

See: skyb_retarget/README.md#the-bone-correspondence
"""

#: Skyblivion skin bone -> Oblivion bone; '[body]' is split by SPLIT_BODY.
SKIN_MAP = {
    'Tail1': 'Bip01 Spine02', 'Tail2': 'Bip01 Spine03', 'Tail3': 'Bip01 Head',
    'FangL[00]': 'Bip01 Tail01',
}
for _side in 'LR':
    SKIN_MAP.update({
        f'Arm{_side}[02]': f'Bip01 {_side}UpperArm',
        f'Arm{_side}Claw': f'Bip01 {_side}ForeArm',
        f'Leg_{_side}[10]': f'Bip01 {_side}Calf1',
        f'Leg_{_side}[11]': f'Bip01 {_side}Foot1',
        f'Leg_{_side}[12]': f'Bip01 {_side}Foot2',
        f'Leg_{_side}[13]': f'Bip01 {_side}Foot3',
        f'Leg_{_side}[14]': f'Bip01 {_side}Foot3',
        f'Leg_{_side}[30]': f'Bip01 {_side}Calf2',
        f'Leg_{_side}[31]': f'Bip01 {_side}Foot4',
        f'Leg_{_side}[32]': f'Bip01 {_side}Foot5',
        f'Leg_{_side}[33]': f'Bip01 {_side}Foot6',
        f'Leg_{_side}[34]': f'Bip01 {_side}Foot6',
    })

#: '[body]' weights go to (rear bone, front bone), blended over this y range.
SPLIT_BODY = ('[body]', 'Bip01 Pelvis', 'Bip01 Spine01', (-5.0, 5.0))

#: Oblivion bone -> where it sits on the Skyblivion model.
LANDMARKS = {
    'Bip01 Spine02': ('boundary', '[body]', 'Tail1'),
    'Bip01 Spine03': ('boundary', 'Tail1', 'Tail2'),
    'Bip01 Neck': ('boundary', 'Tail2', 'Tail3'),
    'Bip01 Head': ('boundary', 'Tail2', 'Tail3'),
    'Bip01 Tail01': ('boundary', '[body]', 'FangL[00]'),
}
#: Feet to plant: ((hip, upper, lower, tip) per leg, body bone) for leg_ik.
LEGS = (tuple((f'Bip01 {s}Thigh{h}', f'Bip01 {s}Foot{a}', f'Bip01 {s}Foot{b}',
               f'Bip01 {s}Foot{b}Nub')
              for s in 'LR' for h, a, b in ((1, 2, 3), (2, 5, 6))),
        'Bip01 NonAccum')

#: The horns over the head: (side sign, side letter); cut from HORN_SOURCE past HORN_SELECT = (|x|, z).
HORNS = ((-1.0, 'L'), (1.0, 'R'))
HORN_SOURCE = 'Tail2'
HORN_SELECT = (5.0, 101.0)


def horn_knots(side: str) -> list:
    """(bone, fraction along the horn): the wing chain off the upper spine."""
    wings = [f'Bip01 {side}Wing0{i}' for i in range(1, 6)]
    return [('Bip01 Spine03', 0.0)] + list(zip(wings, (0.12, 0.30, 0.50, 0.72, 0.92)))


#: (clip, bones) measured from that clip's first frame, not rest: Oblivion arms/wings rest out but idle folded.
REF_POSE = ('idle', [f'Bip01 {s}{b}' for s in 'LR' for b in
                     ('Clavicle', 'UpperArm', 'ForeArm', 'Hand', 'Finger01',
                      'Finger02', 'Finger03', 'Finger04', 'Wing01', 'Wing02',
                      'Wing03', 'Wing04', 'Wing05')])

#: Oblivion bone -> child its segment is turned to point at.
SWING = {}
for _side, _sk in (('L', 'L'), ('R', 'R')):
    LANDMARKS.update({
        f'Bip01 {_side}Clavicle': ('midline', f'Bip01 {_side}UpperArm'),
        f'Bip01 {_side}UpperArm': ('boundary', 'Tail2', f'Arm{_sk}[02]'),
        f'Bip01 {_side}ForeArm': ('boundary', f'Arm{_sk}[02]', f'Arm{_sk}Claw'),
        f'Bip01 {_side}Hand': ('reach', f'Arm{_sk}Claw', 0.75),
        f'Bip01 {_side}Calf1': ('boundary', '[body]', f'Leg_{_sk}[10]'),
        f'Bip01 {_side}Foot1': ('boundary', f'Leg_{_sk}[10]', f'Leg_{_sk}[11]'),
        f'Bip01 {_side}Foot2': ('boundary', f'Leg_{_sk}[11]', f'Leg_{_sk}[12]'),
        f'Bip01 {_side}Foot3': ('boundary', f'Leg_{_sk}[12]', f'Leg_{_sk}[13]'),
        f'Bip01 {_side}Foot3Nub': ('tip', (f'Leg_{_sk}[13]', f'Leg_{_sk}[14]')),
        f'Bip01 {_side}Calf2': ('boundary', '[body]', f'Leg_{_sk}[30]'),
        f'Bip01 {_side}Foot4': ('boundary', f'Leg_{_sk}[30]', f'Leg_{_sk}[31]'),
        f'Bip01 {_side}Foot5': ('boundary', f'Leg_{_sk}[31]', f'Leg_{_sk}[32]'),
        f'Bip01 {_side}Foot6': ('boundary', f'Leg_{_sk}[32]', f'Leg_{_sk}[33]'),
        f'Bip01 {_side}Foot6Nub': ('tip', (f'Leg_{_sk}[33]', f'Leg_{_sk}[34]')),
    })
    _chain = ['Clavicle', 'UpperArm', 'ForeArm', 'Hand']
    _front = ['Thigh1', 'Calf1', 'Foot1', 'Foot2', 'Foot3', 'Foot3Nub']
    _back = ['Thigh2', 'Calf2', 'Foot4', 'Foot5', 'Foot6', 'Foot6Nub']
    _wing = ['Wing01', 'Wing02', 'Wing03', 'Wing04', 'Wing05']
    for _seq in (_chain, _front, _back, _wing):
        for _a, _b in zip(_seq, _seq[1:]):
            SWING[f'Bip01 {_side}{_a}'] = f'Bip01 {_side}{_b}'
