import random

import torch
import numpy as np


SMPL_PARTNAMES = [
   'global', 'head', 'leftCalf', 'leftFingers', 'leftFoot',
   'leftForeArm', 'leftHand', 'leftShoulder', 'leftThigh', 'leftToes',
   'leftUpperArm', 'neck', 'rightCalf', 'rightFingers', 'rightFoot',
   'rightForeArm', 'rightHand', 'rightShoulder', 'rightThigh',
   'rightToes', 'rightUpperArm', 'spine', 'spine1', 'spine2'
]


SMPL_PARTNAMES_MAPPING = {
    # split
    'leftCalf': "left calf",
    'leftFingers': "left fingers",
    'leftFoot': "left foot",
    'leftForeArm': "left forearm",
    'leftHand': "left hand",
    'leftShoulder': "left shoulder",
    'leftThigh': "left thigh",
    'leftToes': "left toes",
    'leftUpperArm': "left upper arm",
    'rightCalf': "right calf",
    'rightFingers': "right fingers",
    'rightFoot': "right foot",
    'rightForeArm': "right forearm",
    'rightHand': "right hand",
    'rightShoulder': "right shoulder",
    'rightThigh': "right thigh",
    'rightToes': "right toes",
    'rightUpperArm': "right upper arm",
    # renamed
    "global": "buttocks",
    "spine": "lower spine",
    "spine1": "spine",
    "spine2": "upper spine",
    # unchanged
    "neck": "neck",
    "head": "head",
}

# templates to generate captions
CAPTION_TEMPLATES = [
    ("{parts} {verb} in contact with {obj_class}", ("is", "are")),
    ("{obj_class} {verb} in contact with {parts}", ("is", )),
    ("{parts} {verb} {obj_class}", ("touches", "touch")),
    ("{obj_class} {verb} {parts}", ("touches", )),
]
CAPTION_TEMPLATES_BASKETBALL = [
    ("A person {verb} {obj_class}", ("is dribbling", )),
] + CAPTION_TEMPLATES
CAPTION_TEMPLATES_SITTING = [
    ("{parts} {verb} on {obj_class}", ("is", "are")),
    ("A person {verb} on {obj_class}", ("is", )),
    ("A person {verb} on {obj_class}", ("sits", )),

] + CAPTION_TEMPLATES
CAPTION_TEMPLATES_HOLDING = [
    ("{obj_class} {verb} in {parts}", ("is", )),
    ("{parts} {verb} {obj_class}", ("holds", "hold")),
    ("{parts} {verb} {obj_class}", ("is grabbing", "are grabbing")),
    ("A person {verb} {obj_class}", ("is holding",)),
    ("A person {verb} {obj_class}", ("is grabbing",)),
    ("A person {verb} {obj_class}", ("is carrying",)),
] + CAPTION_TEMPLATES


# text captions for contacts
"""
<A>,<B>,<C>, and <D> is/are in contact with <obj_class>
A person is interacting with <obj_class> via/by/through <A>,<B>,<C>, and <D>
"""
def contact_to_caption(contact_mask, obj_names, partindex_per_vertex, device):
    B = contact_mask.shape[0]

    part_contacts_sum = torch.zeros((B, 24), dtype=torch.float32, device=device)  # number of vertices in contact for each part
    part_contacts_sum.index_add_(1, partindex_per_vertex, contact_mask)
    part_contacts_mask = part_contacts_sum > 0
    part_contacts_number = part_contacts_mask.sum(dim=-1)
    part_contacts_mask = part_contacts_mask.cpu().numpy()

    captions = []
    for b in range(B):
        obj_name = obj_names[b]

        if obj_name == "basketball":
            caption_template, verbs = random.choice(CAPTION_TEMPLATES_BASKETBALL)
        elif obj_name in ["chairblack", "chairwood", "stool", "yogaball"] and \
                (part_contacts_sum[b][0] > 0 or part_contacts_sum[b][8] > 0 or part_contacts_sum[b][18] > 0):
            caption_template, verbs = random.choice(CAPTION_TEMPLATES_SITTING)
        else:
            caption_template, verbs = random.choice(CAPTION_TEMPLATES_HOLDING)

        if part_contacts_number[b] == 0:
            captions.append(f"A person is next to {obj_name}")
        else:
            if len(verbs) > 1:
                verb = verbs[0] if part_contacts_number[b] == 1 else verbs[1]
            else:
                verb = verbs[0]

            if part_contacts_number[b] == 1:
                parts = SMPL_PARTNAMES_MAPPING[SMPL_PARTNAMES[np.where(part_contacts_mask[b])[0][0]]]
            else:
                parts_list = [
                    SMPL_PARTNAMES_MAPPING[SMPL_PARTNAMES[i]]
                    for i in np.where(part_contacts_mask[b])[0]
                ]
                np.random.shuffle(parts_list)
                # if len(parts_list) > 2:
                #     n_samples = np.random.randint(2, len(parts_list))
                #     parts_list = np.random.choice(parts_list, n_samples, replace=False)
                parts = ", ".join(parts_list[:-1]) + " and " + parts_list[-1]

            captions.append(caption_template.format(parts=parts, verb=verb, obj_class=obj_name))

    return captions
