"""Compile LLM-authored text-to-motion plans to canonical SMPL-X axis angles.

The analytic rig is also an independently labelled proxy for diagnostic Uni-Sign input.
It models a human rest skeleton approximately and does not depend on SMPL-X assets.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FPS = 25
SIDE = {'left': 1, 'right': -1}
FINGERS = ('index', 'middle', 'pinky', 'ring', 'thumb')  # SMPL-X pose order
COCO_FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
ROT_CHANNELS = ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose')
NEUTRAL = (0.24, 0.20, 0.28, 'rest', 'in')  # outward, vertical, forward in meters
POSES = {
    'rest': (.30,.36,.52,.44,.25), 'open': (.05,.06,.08,.07,.17),
    'flat': (.03,.04,.05,.04,.34), 'fist': (1.28,1.30,1.24,1.28,.66),
    'point': (.04,1.30,1.22,1.27,.71), 'v': (.04,.04,1.22,1.27,.71),
    'four': (.03,.03,.03,.03,.78), 'five': (.04,.04,.04,.04,.08),
    'pinch': (.88,1.18,1.20,1.19,.63), 'o': (.84,.86,.89,.85,.74),
    'claw': (.69,.76,.86,.82,.32), 'thumb': (1.19,1.23,1.21,1.22,.05),
    'y': (1.19,1.25,.08,1.21,.06), 'l': (.04,1.23,1.21,1.23,.05),
    'three': (.04,.04,1.20,1.22,.06), 'roll': (.30,1.19,1.20,1.19,.05),
    'horns': (.04,1.25,.08,1.21,.10),
}
# Approximate fingerspelling shapes, with explicit dynamic J/Z paths below.
LETTERS = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (
    'fist flat claw claw fist pinch roll three y y v l fist fist o roll roll v fist fist v v four v y point'
).split()))


def h(x, y, z, hand='open', direction='up'):
    """One signing-space wrist waypoint; positive x is outward on that hand's side."""
    return (x, y, z, hand, direction)


# These are LLM-authored kinematic hypotheses for lexical tokens. Some are
# approximate/iconic, with no dictionary motion, signer data or model feedback.
# Main hand waypoints are executed by the right hand. None keeps the other hand at rest.
LEX = {
    'THIS': ([h(.12,.32,.38,'point','down'), h(.11,.27,.40,'point','down')], None),
    'BEFORE': ([h(.23,.49,.28,'flat','up'), h(.32,.53,.01,'flat','back')], None),
    'YOU': ([h(.11,.43,.29,'point','front'), h(.10,.45,.48,'point','front')], None),
    'ENTER': ([h(.14,.30,.29,'flat','front'), h(.04,.35,.54,'flat','front')],
              [h(.08,.27,.42,'claw','in'), h(.08,.27,.42,'claw','in')]),
    'STAGE': ([h(.08,.26,.43,'flat','out'), h(.41,.26,.42,'flat','out')],
              [h(.08,.26,.43,'flat','out'), h(.41,.26,.42,'flat','out')]),
    'HERE': ([h(.18,.27,.44,'open','up'), h(.24,.21,.46,'open','up'), h(.14,.24,.45,'open','up')],
             [h(.18,.27,.44,'open','up'), h(.24,.21,.46,'open','up'), h(.14,.24,.45,'open','up')]),
    'FOUR': ([h(.26,.52,.31,'four','up')], None),
    'KNEEL': ([h(.14,.39,.37,'v','down'), h(.12,.22,.40,'v','down')],
              [h(.08,.20,.40,'flat','in')]),
    'RELAX': ([h(.08,.46,.23,'open','up'), h(.22,.29,.27,'open','down')],
              [h(.08,.46,.23,'open','up'), h(.22,.29,.27,'open','down')]),
    'PLACE': ([h(.20,.43,.41,'claw','down'), h(.20,.24,.41,'claw','down')], None),
    'LEG': ([h(.18,.18,.29,'open','down'), h(.18,.05,.24,'open','down')],
            [h(.18,.18,.29,'open','down'), h(.18,.05,.24,'open','down')]),
    'THAT': ([h(.19,.39,.28,'point','front'), h(.28,.36,.49,'point','front')], None),
    'ROAD': ([h(.18,.28,.31,'flat','front'), h(.18,.36,.55,'flat','front')],
             [h(.18,.28,.31,'flat','front'), h(.18,.36,.55,'flat','front')]),
    'LONG': ([h(-.14,.32,.36,'point','in'), h(.20,.38,.42,'point','out')],
             [h(.14,.32,.31,'flat','in')]),
    'ALL': ([h(.16,.53,.27,'open','up'), h(.24,.41,.48,'open','in'), h(.03,.28,.37,'flat','in')],
            [h(.09,.29,.37,'flat','in')]),
    'JOIN': ([h(.26,.44,.37,'o','in'), h(.045,.41,.41,'o','in')],
             [h(.26,.44,.37,'o','in'), h(.045,.41,.41,'o','in')]),
    'VERY': ([h(.10,.47,.30,'v','up'), h(.35,.51,.34,'v','up')],
             [h(.10,.47,.30,'v','up'), h(.35,.51,.34,'v','up')]),
    'GOOD': ([h(.04,.57,.22,'flat','up'), h(.09,.30,.41,'flat','up')],
             [h(.10,.26,.36,'flat','in')]),
    'HOW': ([h(.05,.37,.38,'fist','in'), h(.11,.43,.47,'open','up')],
            [h(.05,.37,.38,'fist','in'), h(.11,.43,.47,'open','up')]),
    'ADJUST': ([h(.18,.40,.40,'pinch','up'), h(.23,.45,.41,'pinch','out'), h(.18,.40,.40,'pinch','up')],
               [h(.23,.35,.42,'claw','up')]),
    'STRUM': ([h(.12,.45,.33,'point','in'), h(.10,.31,.37,'point','down'), h(.14,.44,.37,'point','up')],
              [h(.36,.49,.26,'claw','up')]),
    'SEE': ([h(.11,.69,.20,'v','up'), h(.18,.55,.47,'v','front')], None),
    'THERE': ([h(.20,.47,.34,'point','out'), h(.43,.49,.35,'point','out')], None),
    'DRINK': ([h(.14,.45,.35,'claw','up'), h(.055,.57,.27,'claw','in')], None),
    'NAME': ([h(.07,.50,.39,'v','in'), h(.035,.33,.40,'v','in'),
              h(.065,.44,.39,'v','in'), h(.035,.34,.40,'v','in')],
             [h(.07,.30,.41,'v','in')]),
    'GO': ([h(.14,.46,.34,'point','up'), h(.18,.43,.53,'point','front')],
           [h(.14,.46,.34,'point','up'), h(.18,.43,.53,'point','front')]),
    'CREATE': ([h(.06,.43,.38,'fist','in'), h(.02,.38,.38,'fist','down'), h(.08,.44,.38,'fist','up')],
               [h(.06,.33,.37,'fist','in')]),
    'DANCE': ([h(.06,.43,.41,'v','down'), h(-.10,.38,.42,'v','down'), h(.07,.40,.43,'v','down')],
              [h(.05,.30,.39,'flat','in')]),
    'FINISH': ([h(.17,.44,.34,'five','up'), h(.30,.44,.37,'five','out')],
               [h(.17,.44,.34,'five','up'), h(.30,.44,.37,'five','out')]),
    'I': ([h(.11,.40,.32,'point','back'), h(.04,.40,.22,'point','back')], None),
    'HAVE': ([h(.21,.47,.39,'claw','back'), h(.17,.48,.19,'claw','back')],
             [h(.21,.47,.39,'claw','back'), h(.17,.48,.19,'claw','back')]),
    'MITTEN': ([h(.11,.46,.43,'claw','in'), h(-.05,.39,.38,'claw','in')],
               [h(.08,.42,.40,'flat','up')]),
    'DIFFERENT': ([h(.05,.40,.38,'point','in'), h(.30,.41,.40,'point','out')],
                  [h(.05,.40,.35,'point','in'), h(.30,.41,.40,'point','out')]),
    'MAKE': ([h(.045,.46,.37,'fist','in'), h(.045,.37,.39,'fist','down'), h(.08,.39,.36,'fist','back')],
             [h(.05,.32,.37,'fist','in')]),
    'OK': ([h(.23,.53,.33,'o','up'), h(.27,.50,.36,'o','up')], None),
    'WE': ([h(.17,.45,.29,'point','back'), h(-.16,.45,.28,'point','back')], None),
    'SHOW': ([h(.025,.44,.37,'point','up'), h(.08,.46,.53,'point','up')],
             [h(.025,.44,.33,'flat','up'), h(.08,.46,.49,'flat','up')]),
    'POSITION': ([h(.20,.44,.43,'claw','down'), h(.20,.28,.43,'claw','down')],
                 [h(.20,.44,.43,'claw','down'), h(.20,.28,.43,'claw','down')]),
    'DISCUSS': ([h(.09,.43,.38,'point','in'), h(.12,.38,.41,'point','in'), h(.08,.43,.38,'point','in')],
                [h(.10,.29,.41,'flat','in')]),
    'REVERSE': ([h(.07,.43,.40,'point','in'), h(.18,.50,.41,'point','up'), h(.08,.37,.42,'point','out')],
                [h(.08,.35,.40,'point','in'), h(.17,.38,.45,'point','up'), h(.07,.49,.44,'point','out')]),
    'BASKETBALL': ([h(.18,.43,.45,'claw','in'), h(.18,.65,.50,'open','up')],
                   [h(.18,.43,.45,'claw','in'), h(.18,.65,.50,'open','up')]),
    'BREATHE': ([h(.15,.33,.19,'claw','back'), h(.21,.56,.28,'open','up')],
                [h(.15,.33,.19,'claw','back'), h(.21,.56,.28,'open','up')]),
    'UP': ([h(.22,.31,.39,'point','up'), h(.22,.70,.36,'point','up')], None),
    'BECAUSE': ([h(.14,.67,.18,'l','up'), h(.23,.58,.30,'fist','up')], None),
    'BED': ([h(-.10,.47,.32,'flat','up'), h(-.09,.59,.21,'flat','up')],
            [h(.10,.45,.30,'flat','up'), h(.10,.56,.19,'flat','up')]),
    'YOUR': ([h(.12,.42,.34,'flat','up'), h(.12,.42,.49,'flat','up')], None),
    'PILL': ([h(.12,.43,.38,'pinch','up'), h(.055,.56,.29,'pinch','in')], None),
    'NEED': ([h(.19,.45,.35,'pinch','down'), h(.19,.31,.39,'pinch','down')], None),
    'NOT': ([h(.05,.56,.24,'thumb','up'), h(.18,.51,.49,'thumb','up')], None),
    'BIG': ([h(.08,.46,.39,'claw','in'), h(.40,.46,.42,'claw','in')],
            [h(.08,.46,.39,'claw','in'), h(.40,.46,.42,'claw','in')]),
    'STEP': ([h(.12,.33,.31,'flat','front'), h(.13,.44,.41,'flat','front'), h(.13,.31,.51,'flat','front')],
             [h(.12,.33,.34,'flat','front'), h(.12,.34,.28,'flat','front'), h(.13,.43,.45,'flat','front')]),
    'DOWN': ([h(.22,.56,.39,'point','down'), h(.22,.21,.39,'point','down')], None),
    'MAYBE': ([h(.19,.27,.39,'flat','up'), h(.19,.43,.39,'flat','up'), h(.19,.28,.39,'flat','up')],
              [h(.19,.43,.39,'flat','up'), h(.19,.27,.39,'flat','up'), h(.19,.43,.39,'flat','up')]),
    'GOLD': ([h(.19,.64,.17,'point','up'), h(.32,.49,.36,'y','up')], None),
    'OR': ([h(.21,.37,.35,'o','up'), h(.25,.46,.34,'v','up')], None),
    'WHITE': ([h(.10,.46,.20,'five','back'), h(.14,.45,.43,'pinch','up')], None),
    'ONE': ([h(.21,.50,.37,'point','up')], None),
    'TYPE': ([h(.08,.44,.38,'claw','in'), h(.19,.42,.43,'claw','in')],
             [h(.08,.34,.38,'claw','in'), h(.19,.35,.43,'claw','in')]),
    'MEASURE': ([h(.07,.39,.39,'y','in'), h(.30,.39,.40,'y','out')],
                [h(.07,.39,.39,'y','in'), h(.28,.39,.40,'y','out')]),
    'SHOOT': ([h(.12,.41,.36,'l','front'), h(.12,.43,.52,'l','front'), h(.14,.49,.45,'l','front')],
              [h(.04,.38,.47,'claw','front')]),
    'FAST': ([h(.10,.38,.35,'l','in'), h(.34,.48,.41,'l','out')],
             [h(.10,.38,.35,'l','in'), h(.34,.48,.41,'l','out')]),
    'KNOW': ([h(.12,.69,.22,'flat','up'), h(.15,.64,.23,'flat','up')], None),
    'WHAT': ([h(.15,.34,.42,'open','up'), h(.27,.35,.42,'open','up'), h(.14,.33,.42,'open','up')],
             [h(.15,.34,.42,'open','up'), h(.27,.35,.42,'open','up'), h(.14,.33,.42,'open','up')]),
}
DIR = {'up': (0,.97,.22), 'down': (0,-.97,.22), 'front': (0,.20,.98),
       'back': (0,.05,-1), 'in': (-.98,.15,.09), 'out': (.98,.15,.09)}


def unit(v):
    return v / np.linalg.norm(v)


def basis(axis, secondary):
    axis = unit(np.asarray(axis))
    second = unit(np.asarray(secondary) - np.dot(axis, secondary) * axis)
    return np.stack((axis, second, np.cross(axis, second)), axis=1)


def align(axis, second, target, target_second):
    return Rotation.from_matrix(basis(target, target_second) @ basis(axis, second).T)


def arm(side, waypoint):
    """Two-bone IK in signing space, then analytical local shoulder, elbow, wrist rotations."""
    s = SIDE[side]
    shoulder = np.array([s*.18, .46, 0.])
    offset_u = np.array([s*.285,-.018,-.006])
    offset_f = np.array([s*.255,-.012,0.])
    upper, lower = np.linalg.norm(offset_u), np.linalg.norm(offset_f)
    x,y,z,shape,direction = waypoint
    delta = np.array([s*x,y,z]) - shoulder
    d = np.clip(np.linalg.norm(delta), .10, upper + lower - .010)
    axis = unit(delta)
    elbow_hint = np.array([s*.47,-1.,-.12])
    bend = unit(elbow_hint - np.dot(elbow_hint, axis)*axis)
    along = (d*d + upper*upper - lower*lower)/(2*d)
    upper_dir = along*axis + np.sqrt(max(0,upper*upper-along*along))*bend
    fore_dir = d*axis-upper_dir
    sh = align(offset_u, [0,-1,0], upper_dir, fore_dir)
    world_el = align(offset_f, [0,-1,0], fore_dir, -upper_dir)
    fingers = np.asarray(DIR[direction]) * [s,1,1]
    # Point 'up' with palm approximately camera-facing, point 'front' with palm toward floor.
    palm = (0,-1,0) if direction in ('front','back') else (0,0,1)
    world_wr = align([s,0,0], [0,-1,0], fingers, palm)
    return sh, sh.inv()*world_el, world_el.inv()*world_wr, hand_shape(side, shape)


def hand_shape(side, shape):
    s = SIDE[side]
    curls = np.asarray(POSES[shape])
    joints = np.zeros((5,3,3))
    joints[:,:,2] = (-s)*curls[:,None]*[.78,.85,.52]
    joints[:,0,1] = np.array([.10,.00,-.16,-.08,.19]) * s
    if shape in ('v','four','five','three','horns','open'):
        joints[:,0,1] *= 1.55
    joints[-1,0,0] = .17*s
    return Rotation.from_euler('XYZ', joints.reshape(15,3)).as_rotvec().ravel()


def pose_frame(r, l, head):
    body = np.zeros((21,3), np.float32)
    for side, waypoint, inds in [('left', l, (15,17,19)), ('right', r, (16,18,20))]:
        sh, el, wrist, fingers = arm(side, waypoint)
        for i, rot in zip(inds, (sh,el,wrist)):
            body[i] = rot.as_rotvec()
        if side == 'left':
            left = fingers
        else:
            right = fingers
    body[11,0] = head * .35
    body[14,0] = head * .65
    return np.concatenate((body.ravel(), left, right))


def paired_steps(token):
    if token.startswith('FS:'):
        pts = []
        for i, letter in enumerate(token[3:]):
            x = .21+.022*(i%4)
            if letter == 'J':
                path = [h(x,.52,.37,'y'), h(x,.47,.39,'y','down'), h(x+.06,.49,.39,'y','out')]
            elif letter == 'Z':
                path = [h(x,.53,.37,'point'), h(x+.07,.53,.37,'point','out'),
                        h(x,.46,.38,'point','in'), h(x+.07,.46,.38,'point','out')]
            else:
                path = [h(x,.50,.37,LETTERS[letter])]
            pts.extend((right, NEUTRAL, 4 if len(path)==1 else 3) for right in path)
        return pts
    rh, lh = LEX[token]
    lh = lh or [NEUTRAL]
    count = max(len(rh),len(lh))
    return [(rh[min(i,len(rh)-1)],lh[min(i,len(lh)-1)], 8 if count==1 else 6)
            for i in range(count)]


def interpolate(a, b, frames):
    t = np.arange(1,frames+1) / frames
    t = t*t*(3-2*t)
    v = []
    for start,end in zip(a.reshape(-1,3),b.reshape(-1,3)):
        v.append(Slerp([0,1],Rotation.from_rotvec([start,end]))(t).as_rotvec())
    return np.stack(v,axis=1).reshape(frames,-1)


def sequence(tokens):
    start = pose_frame(NEUTRAL, NEUTRAL, 0)
    blocks = [np.repeat(start[None],5,0)]
    prev = start
    for token in tokens:
        rotation = -.12 if token in ('HOW','WHAT') else .08 if token in ('GOOD','FINISH') else 0.
        for right,left,frames in paired_steps(token):
            current = pose_frame(right,left,rotation)
            blocks.append(interpolate(prev,current,frames))
            prev = current
        # Two-frame hold preserves a lexical target even after smooth interpolation.
        blocks.append(np.repeat(prev[None],2,0))
    blocks.append(interpolate(prev,start,7))
    blocks.append(np.repeat(start[None],4,0))
    rotations = np.concatenate(blocks).astype('float32')
    n = len(rotations)
    return dict(global_orient=np.zeros((n,3),'float32'), body_pose=rotations[:,:63],
                left_hand_pose=rotations[:,63:108], right_hand_pose=rotations[:,108:153],
                jaw_pose=np.zeros((n,3),'float32'), expression=np.zeros((n,10),'float32'),
                transl=np.zeros((n,3),'float32'), betas=np.zeros(10,'float32'), fps=np.float32(FPS))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans',type=Path,default=Path(__file__).with_name('plans.json'))
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('poses'))
    parser.add_argument('--stats',type=Path)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for clip_id, tokens in json.loads(args.plans.read_text()).items():
        pose=sequence(tokens)
        np.savez_compressed(args.out/f'{clip_id}.npz',**pose)
        n=len(pose['body_pose'])
        rows.append(dict(id=clip_id,frames=n,fps=FPS,duration_s=n/FPS,tokens=len(tokens),
                         fingerspelled=sum(t.startswith('FS:') for t in tokens)))
    if args.stats:
        args.stats.parent.mkdir(exist_ok=True,parents=True)
        with args.stats.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__=='__main__':
    main()
