import pickle as pkl

import clip
import numpy as np
import torch

from tridi.model.conditioning.contact_ae_clip import ContactEnc, ContactEncCLIP, ContactDec
from tridi.utils.contacts import contact_to_caption


class ContactModel:
    """
    A class that wraps contact encoding and decoding models.
    """
    def __init__(self, contact_model_type, device='cpu') -> None:
        self.contact_model_type = contact_model_type

        if not contact_model_type in ["encoder_decimated_clip", "NONE"]:
            raise ValueError(f"Unsupported contact conditioning {contact_model_type}")

        self.clip_model = None
        self.contact_ae = None
        self.contact_enc = None
        self.contact_dec = None

        self.device = device

    def set_contact_model(self, model_type, enc_dec_type, weights_path) -> None:
        if model_type == "NONE":
            return None

        if enc_dec_type == "model_enc":
            contact_ae = ContactEnc(690, 256, 128)
        elif enc_dec_type == "model_dec":
            contact_ae = ContactDec(690, 256, 128)
        elif enc_dec_type == "model_enc_clip":
            contact_ae = ContactEncCLIP(512, 256, 128)
            self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
            self.clip_model = self.clip_model.eval()

            # load partindexes for sampling captions
            smpl_template_idxs = np.load("./assets/smpl_template_decimated_idxs.npy")
            with open("./assets/smpl_segmentation.pkl", "rb") as fp:
                partindex_per_vertex = pkl.load(fp)['vertex_to_partindex']
            decimated_partindex_per_vertex = partindex_per_vertex[smpl_template_idxs]
            self.smpl_template_idxs_th = \
                torch.tensor(smpl_template_idxs, dtype=torch.long, device=self.device)
            self.decimated_partindex_per_vertex_th = \
                torch.tensor(decimated_partindex_per_vertex, dtype=torch.long, device=self.device)

        contact_ae = contact_ae.to("cuda")
        checkpoint = torch.load(weights_path, map_location='cuda', weights_only=False)
        state_dict = checkpoint[enc_dec_type]
        missing_keys, unexpected_keys = contact_ae.load_state_dict(state_dict, strict=True)
        contact_ae = torch.compile(contact_ae, mode="reduce-overhead")
        contact_ae = contact_ae.eval()

        if enc_dec_type == "model_dec":
            self.contact_dec = contact_ae
        else:
            self.contact_enc = contact_ae

    def encode_contacts(
            self, sbj_vertices, sbj_subindexes, obj_keypoints,
            obj_class, obj_R, enc_type="", contact_thr=0.05, obj_name=None
    ):
        captions = []
        B = sbj_vertices.shape[0]
        sbj_contacts, sbj_contacts_full = None, None

        if self.contact_model_type in ["encoder_decimated_clip", "NONE"]:
            contact_vertices = sbj_vertices[:, sbj_subindexes[0]]
        else:
            raise ValueError(f"Unsupported contact conditioning {self.contact_model_type}")

        sbj2obj_contacts_distances = torch.cdist(
            contact_vertices,
            obj_keypoints.to(contact_vertices.device)
        ).min(dim=-1)
        sbj_contacts_distances = sbj2obj_contacts_distances.values.contiguous().float()  # .to(contact_vertices.device)

        # 5 cm thr for contact
        sbj_contacts_mask = (sbj_contacts_distances < contact_thr)
        sbj_contacts_mask = sbj_contacts_mask.float()

        if self.contact_model_type == "NONE":
            sbj_contacts = None
            sbj_contacts_full = sbj_contacts_distances
        elif self.contact_model_type == "encoder_decimated_clip":
            if enc_type == "model_enc":
                sbj_contacts = self.contact_enc(sbj_contacts_mask).float() + 0  # weird bug in torch.compile
            elif enc_type == "model_enc_clip":
                # convert contacts to text and sample from text
                captions = contact_to_caption(
                    sbj_contacts_mask, obj_name, self.decimated_partindex_per_vertex_th, self.device
                )

                tokenized_captions = clip.tokenize(captions).to(self.device)
                clip_features = self.clip_model.encode_text(tokenized_captions).float()
                sbj_contacts = self.contact_enc(clip_features).float() + 0  # weird bug in torch.compile

            sbj_contacts_full = sbj_contacts_distances
        else:
            raise ValueError(f"Unsupported contact conditioning {self.contact_model_type}")

        return sbj_contacts, sbj_contacts_full, captions

    def decode_contacts_th(
            self, gt_contacts, pred_contacts, is_sampling_contacts, gt_contact_thr=0.05, pred_contact_thr=0.4
    ):
        if self.contact_model_type == "encoder_decimated_clip":
            if is_sampling_contacts:  # sample contacts
                pred_contacts_dec = self.contact_dec(pred_contacts) + 0  # weird bug in torch.compile
                contacts = pred_contacts_dec
                contacts_mask_dec = pred_contacts_dec > pred_contact_thr
            else:  # condition on contacts
                contacts = gt_contacts
                contacts_mask_dec = (gt_contacts < gt_contact_thr)
        else:
            raise ValueError(f"Unsupported contact conditioning {self.contact_model_type}")

        return contacts, contacts_mask_dec

    @torch.no_grad()
    def decode_contacts_np(
            self, gt_contacts, pred_contacts, sbj_subindexes, is_sampling_contacts
    ):
        B = pred_contacts.shape[0] if is_sampling_contacts else gt_contacts.shape[0]

        _, contacts_mask_dec = self.decode_contacts_th(
            gt_contacts, pred_contacts, is_sampling_contacts
        )
        contacts_mask_dec = contacts_mask_dec.cpu().numpy()
        contacts_mask = np.zeros((B, 6890), dtype=bool)
        sbj_contact_indexes = sbj_subindexes[0]

        contacts_mask[:, sbj_contact_indexes] = contacts_mask_dec

        return contacts_mask
