from portal2vroid.bonemap import classify_source_bones, map_vrm_to_source


def _tree(spec):
    names = [n for n, _ in spec]
    parents = [names.index(p) if p else -1 for _, p in spec]
    return names, parents


def test_tf2_style_names():
    spec = [("bip_pelvis", None), ("bip_spine_0", "bip_pelvis"), ("bip_spine_1", "bip_spine_0"), ("bip_spine_2", "bip_spine_1"),
            ("bip_spine_3", "bip_spine_2"), ("bip_neck", "bip_spine_3"), ("bip_head", "bip_neck")]
    for s in ("L", "R"):
        spec += [
            (f"bip_collar_{s}", "bip_spine_3"), (f"bip_upperArm_{s}", f"bip_collar_{s}"), (f"bip_lowerArm_{s}", f"bip_upperArm_{s}"),
            (f"bip_hand_{s}", f"bip_lowerArm_{s}"), (f"bip_thumb_0_{s}", f"bip_hand_{s}"), (f"bip_thumb_1_{s}", f"bip_thumb_0_{s}"),
            (f"bip_thumb_2_{s}", f"bip_thumb_1_{s}"), (f"bip_index_0_{s}", f"bip_hand_{s}"), (f"bip_index_1_{s}", f"bip_index_0_{s}"),
            (f"bip_hip_{s}", "bip_pelvis"), (f"bip_knee_{s}", f"bip_hip_{s}"), (f"bip_foot_{s}", f"bip_knee_{s}"), (f"bip_toe_{s}", f"bip_foot_{s}"),
            (f"weapon_bone_{s}", f"bip_hand_{s}"),
        ]
    names, parents = _tree(spec)
    slots, spines = classify_source_bones(names, parents)
    n = lambda slot: names[slots[slot]]
    assert n("hips") == "bip_pelvis" and n("head") == "bip_head" and n("neck") == "bip_neck"
    assert n("leftUpperLeg") == "bip_hip_L" and n("leftLowerLeg") == "bip_knee_L" and n("rightFoot") == "bip_foot_R"
    assert n("leftToes") == "bip_toe_L"
    assert n("leftShoulder") == "bip_collar_L" and n("rightUpperArm") == "bip_upperArm_R" and n("leftHand") == "bip_hand_L"
    assert n("leftThumbMetacarpal") == "bip_thumb_0_L" and n("leftThumbDistal") == "bip_thumb_2_L"
    assert n("rightIndexIntermediate") == "bip_index_1_R"
    assert [names[i] for i in spines] == ["bip_spine_0", "bip_spine_1", "bip_spine_2", "bip_spine_3"]
    vrm = {"hips": 0, "spine": 1, "chest": 2, "upperChest": 3, "leftUpperLeg": 9}
    m = map_vrm_to_source(vrm, slots, spines)
    assert names[m["spine"]] == "bip_spine_0" and names[m["upperChest"]] == "bip_spine_3"
    assert names[m["chest"]] in ("bip_spine_1", "bip_spine_2")


def test_valvebiped_helpers_are_ignored():
    vb = "ValveBiped.Bip01_"
    spec = [(vb + "Pelvis", None), (vb + "Spine", vb + "Pelvis"), (vb + "Spine4", vb + "Spine"), (vb + "Head1", vb + "Spine4"),
            (vb + "L_Clavicle", vb + "Spine4"), (vb + "L_Shoulder", vb + "Spine4"), (vb + "L_UpperArm", vb + "L_Clavicle"),
            (vb + "L_Bicep", vb + "L_UpperArm"), (vb + "L_Forearm", vb + "L_UpperArm"), (vb + "L_Elbow", vb + "L_UpperArm"),
            (vb + "L_Ulna", vb + "L_Forearm"), (vb + "L_Wrist", vb + "L_Forearm"), (vb + "L_Hand", vb + "L_Forearm"),
            (vb + "L_Thigh", vb + "Pelvis"), (vb + "L_Calf", vb + "L_Thigh"), (vb + "L_Foot", vb + "L_Calf")]
    names, parents = _tree(spec)
    slots, spines = classify_source_bones(names, parents)
    n = lambda slot: names[slots[slot]]
    assert n("leftShoulder") == vb + "L_Clavicle"
    assert n("leftLowerArm") == vb + "L_Forearm" and n("leftHand") == vb + "L_Hand"
    assert n("leftLowerLeg") == vb + "L_Calf"
    assert [names[i] for i in spines] == [vb + "Spine", vb + "Spine4"]
    vrm = {"spine": 1, "chest": 2, "upperChest": 3}
    m = map_vrm_to_source(vrm, slots, spines)
    # only two Source spine bones for three VRM ones: first and last map, the middle stays unmapped
    assert names[m["spine"]] == vb + "Spine" and names[m["upperChest"]] == vb + "Spine4" and "chest" not in m
