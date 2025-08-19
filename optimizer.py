import os
import sys
import random
from tqdm import tqdm
import copy
import pandas as pd
import ast
from collections import defaultdict
from pathlib import Path
import json
import math
random.seed(42)  # For reproducibility

def _normalize_features(val):
    """Return a set of feature tokens from val (string/list/tuple/set/NaN/None)."""
    if val is None:
        return set()
    if isinstance(val, float) and math.isnan(val):
        return set()
    if isinstance(val, (list, tuple, set)):
        return {str(x).strip() for x in val if str(x).strip()}
    # treat everything else as string (including numbers)
    s = str(val)
    if s.lower() == 'nan':
        return set()
    # split on commas; tweak if your CSV uses another delimiter
    return {tok.strip() for tok in s.split(',') if tok.strip()}

def _room_meets_required_feature(room_id, required_feature, room_dict):
    """True if the room has ALL required features (single or comma-separated)."""
    req = _normalize_features(required_feature)
    if not req:
        return True
    room_feats = _normalize_features(room_dict.get(room_id, {}).get('features', ''))
    return req.issubset(room_feats)

def _room_is_free_for_time(room_id, time_block, time_to_room, time_block_overlap_dict):
    """Room is free if it's unused at the candidate time block OR any overlapping block."""
    occupied = set(time_to_room.get(time_block, set()))
    for t in time_block_overlap_dict.get(time_block, []):
        occupied |= set(time_to_room.get(t, set()))
    return room_id not in occupied

def _prof_is_free_for_time(section_id, new_time_block, time_to_sec, time_block_overlap_dict, prof_pref_dict):
    prof_id = prof_pref_dict[section_id].get('professor_id')
    if not prof_id:
        return True
    
    times = set(time_block_overlap_dict.get(new_time_block, []))
    for t in times:
        for other in time_to_sec.get(t, set()):
            if other == section_id:
                continue
            if prof_pref_dict[other].get('professor_id') == prof_id:
                return False
    return True

def _enrolled_and_non_enrolled(section_dict, prof_pref_dict):
    enrolled = set(section_dict.keys())
    all_secs = set(prof_pref_dict.keys())
    non_enrolled = list(all_secs - enrolled)
    enrolled = list(enrolled)
    return enrolled, non_enrolled

def _choose_slots_for(section_id, time_block_dict, time_block_overlap_dict,
                      room_dict, prof_pref_dict, time_to_sec, time_to_room,
                      prefer_strict=True):
    """
    Try to pick (time_block, room) for section_id.
    prefer_strict=True -> try preferred rooms at preferred times first.
    Falls back to any feature-compliant room at preferred times, then allowed-but-not-preferred times.
    Returns (tb, room) or None.
    """
    meta = prof_pref_dict[section_id]
    tp = meta['time_block_type']
    impossible = set(meta.get('impossible_time_blocks') or [])
    pref_times_raw = meta.get('preferred_time_blocks') or []
    pref_times = [tb for tb in pref_times_raw if tb not in impossible]
    other_times = [tb for tb in time_block_dict.get(tp, []) if tb not in impossible and tb not in pref_times]

    required_feature = meta.get('required_room_feature', '')
    pref_rooms = meta.get('preferred_room_ids') or []
    all_rooms = list(room_dict.keys())
    feature_ok_rooms = [r for r in all_rooms if _room_meets_required_feature(r, required_feature, room_dict)]

    # order rooms: preferred first, then other feature-compliant
    room_order = [r for r in pref_rooms if r in feature_ok_rooms] + [r for r in feature_ok_rooms if r not in pref_rooms]

    # 1) preferred rooms @ preferred times (if prefer_strict)
    if prefer_strict:
        for r in [rr for rr in pref_rooms if rr in feature_ok_rooms]:
            for tb in pref_times:
                if _room_is_free_for_time(r, tb, time_to_room, time_block_overlap_dict) and \
                   _prof_is_free_for_time(section_id, tb, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                    return tb, r

    # 2) any feature-ok room @ preferred times
    for r in room_order:
        for tb in pref_times:
            if _room_is_free_for_time(r, tb, time_to_room, time_block_overlap_dict) and \
               _prof_is_free_for_time(section_id, tb, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                return tb, r

    # 3) any feature-ok room @ allowed-but-not-preferred times
    for r in room_order:
        for tb in other_times:
            if _room_is_free_for_time(r, tb, time_to_room, time_block_overlap_dict) and \
               _prof_is_free_for_time(section_id, tb, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                return tb, r

    return None

prof_busy_blocks = defaultdict(set)  # prof_id -> set(blocks)
room_busy_blocks = defaultdict(set)  # room_id -> set(blocks)

def _mark_occupancy_for_assignment(section_id, tb, rm, prof_pref_dict, time_block_overlap_dict):
    pid = prof_pref_dict[section_id].get('professor_id')
    clos = set(time_block_overlap_dict.get(tb, []))  # closure(tb) includes tb itself in your code
    if pid:
        for u in clos:
            prof_busy_blocks[pid].add(u)
    for u in clos:
        room_busy_blocks[rm].add(u)

def _unmark_occupancy_for_assignment(section_id, tb, rm, prof_pref_dict, time_block_overlap_dict):
    pid = prof_pref_dict[section_id].get('professor_id')
    clos = set(time_block_overlap_dict.get(tb, []))
    if pid:
        for u in clos:
            prof_busy_blocks[pid].discard(u)
    for u in clos:
        room_busy_blocks[rm].discard(u)

def _deepcopy_sets_dict(d):
    return {k: set(v) for k, v in d.items()}

def _snapshot_state(time_to_sec, time_to_room, sec_to_time, sec_to_room,
                    prof_busy_blocks, room_busy_blocks):
    return {
        "time_to_sec": _deepcopy_sets_dict(time_to_sec),
        "time_to_room": _deepcopy_sets_dict(time_to_room),
        "sec_to_time": dict(sec_to_time),
        "sec_to_room": dict(sec_to_room),
        "prof_busy_blocks": _deepcopy_sets_dict(prof_busy_blocks),
        "room_busy_blocks": _deepcopy_sets_dict(room_busy_blocks),
    }

def _restore_state(snap, time_to_sec, time_to_room, sec_to_time, sec_to_room,
                   prof_busy_blocks, room_busy_blocks):
    # mutate in place to preserve external references
    time_to_sec.clear();  time_to_sec.update(_deepcopy_sets_dict(snap["time_to_sec"]))
    time_to_room.clear(); time_to_room.update(_deepcopy_sets_dict(snap["time_to_room"]))
    sec_to_time.clear();  sec_to_time.update(snap["sec_to_time"])
    sec_to_room.clear();  sec_to_room.update(snap["sec_to_room"])

    prof_busy_blocks.clear()
    for k, v in snap["prof_busy_blocks"].items():
        prof_busy_blocks[k] = set(v)

    room_busy_blocks.clear()
    for k, v in snap["room_busy_blocks"].items():
        room_busy_blocks[k] = set(v)

def _rebuild_occupancy_from_assignments(sec_to_time, sec_to_room, prof_pref_dict, time_block_overlap_dict):
    prof_busy_blocks.clear()
    room_busy_blocks.clear()
    for s, tb in sec_to_time.items():
        rm = sec_to_room.get(s)
        if tb and rm:
            _mark_occupancy_for_assignment(s, tb, rm, prof_pref_dict, time_block_overlap_dict)

# Helper functions
#########################################################

def compute_time_block_overlap_map(df_timeblocks):

    df_timeblocks['time'] = df_timeblocks['time'].apply(
        lambda x: ast.literal_eval(str(x).replace("“", "\"").replace("”", "\""))
    )
    time_block_overlap_map = {}
    block_metadata = {}

    for i, row_a in df_timeblocks.iterrows():
        block_metadata[row_a['block_id']] = row_a['time']
        block_a = row_a['block_id']
        time_a = row_a['time']
        for j, row_b in df_timeblocks.iterrows():
            if i >= j:
                continue
            block_b = row_b['block_id']
            if block_a == block_b:
                continue
            time_b = row_b['time']

            overlap = False
            for start_a, end_a, day_a in time_a:
                for start_b, end_b, day_b in time_b:
                    if day_a == day_b:
                        start_a_min = int(start_a.split(':')[0]) * 60 + int(start_a.split(':')[1]) - 480
                        end_a_min   = int(end_a.split(':')[0]) * 60 + int(end_a.split(':')[1]) - 480
                        start_b_min = int(start_b.split(':')[0]) * 60 + int(start_b.split(':')[1]) - 480
                        end_b_min   = int(end_b.split(':')[0]) * 60 + int(end_b.split(':')[1]) - 480
                        if not (end_a_min <= start_b_min or end_b_min <= start_a_min):
                            overlap = True
                            break
                if overlap:
                    break

            time_block_overlap_map[(block_a, block_b)] = overlap
            time_block_overlap_map[(block_b, block_a)] = overlap

    # Build overlap dict without duplicates
    time_block_overlap_dict = defaultdict(set)
    for (block_a, block_b), overlap in time_block_overlap_map.items():
        time_block_overlap_dict[block_a].add(block_a)
        if overlap:
            time_block_overlap_dict[block_a].add(block_b)
            time_block_overlap_dict[block_b].add(block_a)

    # Convert sets to lists
    time_block_overlap_dict = {k: list(v) for k, v in time_block_overlap_dict.items()}

    return time_block_overlap_map, block_metadata, time_block_overlap_dict

def create_class_section_dict(df_course_offered):
    # Output: Dictionsay | Key: class_id, Value: list of section_ids
    # df_course_offered has columns: [Section ID | Course ID | Type | Time_block_type | Prof ID | Prof Name | Expected Enrollment | No conflict course ID | Multi-section]
    class_section_dict = {}
    for index, row in df_course_offered.iterrows():
        if index == 0:
            continue
        section_id = row['Section ID']
        class_id = section_id.split('^')[0]  # Extract course ID from section ID
        if class_id not in class_section_dict:
            class_section_dict[class_id] = []
        class_section_dict[class_id].append(section_id)
    return class_section_dict

def create_multi_section_dict(df_course_offered): 
    # Output: Dictionary | Key: class_id, Value: list of section_ids for multisection courses
    multisection_classes = {}
    counter = 0
    for index, row in df_course_offered.iterrows():
        if index == 0:
            continue
        section_id = row['Section ID']
        class_id = section_id.split('^')[0]  # Extract course ID from section ID
        if row['Multi-section'] == 'TRUE' or row['Multi-section'] == True or row['Multi-section'] == "TRUE":
            if class_id not in multisection_classes:
                multisection_classes[class_id] = []
            multisection_classes[class_id].append(section_id)
            counter = counter + 1
    print(f"Number of multisection courses: {counter}")
    return multisection_classes 

def read_time_block_dict(df_time_block_dict):
    time_block_dict = {}
    for index, row in df_time_block_dict.iterrows():
        time_block_type = row['Time_block_type']
        try:
            raw = str(row['block_ids']).replace("“","\"").replace("”","\"")
            block_ids = ast.literal_eval(raw)
            if isinstance(block_ids, list):
                time_block_dict[time_block_type] = [bid.strip().strip('"').strip("'") for bid in block_ids]
            else:
                print(f"Warning: block_ids for {time_block_type} is not a list: {block_ids}")
        except Exception as e:
            print(f" Failed to parse block_ids for {time_block_type}: {e}")
    return time_block_dict

def build_prof_pref_dict(df_prof_prefs):

    def clean_string_list(l):
        # Remove extra quotes and whitespace from each item in list
        return [s.strip().strip('"').strip("'") for s in l if isinstance(s, str)]
    prof_pref_dict = {}
    for index, row in df_prof_prefs.iterrows():
        if index == 0:
            continue
        section_id = row['Section ID']
        if section_id not in prof_pref_dict:
            try:
                impossible = ast.literal_eval(row['Impossible Time Blocks'])
                preferred = ast.literal_eval(row['Preferred Time Blocks'])
                preferred_rooms = ast.literal_eval(row['Preferred Room IDs'])
            except Exception as e:
                print(f"Error parsing list for section {section_id}: {e}")
                impossible = []
                preferred = []
                preferred_rooms = []

            prof_pref_dict[section_id] = {
                'name': row['Professor Name'],
                'professor_id': row['Professor ID'],
                'class_id': row['Class ID'],
                'b2b_preference': row['B2B Preference'],
                'time_block_type': row['Time_block_type'],
                'impossible_time_blocks': clean_string_list(impossible),
                'preferred_time_blocks': clean_string_list(preferred),
                'required_room_feature': row['Required Room Feature'],
                'preferred_room_ids': clean_string_list(preferred_rooms),
            }
    return prof_pref_dict

def build_room_dict(df_rooms):
    # Build a dictionary to map room IDs to their properties
    # df_rooms has columns: ['Room ID', 'Building', 'Room Capacity', 'Room Features']
    room_dict = {}
    for index, row in df_rooms.iterrows():
        if index == 0:
            continue
        room_id = row['Full Room ID']
        room_dict[room_id] = {
            'building': row['Building Name'],
            'capacity': row['Room Capacity'],
            'features': row['Room Features']
        }
    return room_dict

#########################################################

def student_pref_to_section_dict(student_pref_section_dict, prof_pref_dict):
    # Convert student_pref_section_dict to a section-based dictionary
    # Key: section_id, Value: list of student_ids
    section_dict = {}
    for student_id, section_ids in student_pref_section_dict.items():
        for section_id in section_ids:
            if section_id not in section_dict:
                section_dict[section_id] = []
            section_dict[section_id].append(student_id)

    for section_id in prof_pref_dict.keys():
        if section_id not in section_dict:
            section_dict[section_id] = []
    return section_dict

def build_adjacency_matrix(section_dict):
    # Build an adjacency matrix for the sections based on shared students
    # section_dict has Keys: section ID, Items: list of students signed up for that class
    section_ids = list(section_dict.keys())
    n = len(section_ids)
    adjacency_matrix = [[0] * n for _ in range(n)]
    adjacency_matrix_order_map = [] 
    
    for i in range(n):
        section_id = section_ids[i]
        adjacency_matrix_order_map.append(section_id)  # Initialize a map for each section
        for j in range(i + 1, n):
            shared_students = set(section_dict[section_ids[i]]) & set(section_dict[section_ids[j]])
            adjacency_matrix[i][j] = len(shared_students)
            adjacency_matrix[j][i] = len(shared_students)  # Symmetric matrix
    
    return adjacency_matrix, adjacency_matrix_order_map

#########################################################

def update_student_preferences_for_single_section_classes(df_student_preferences, class_section_dict):
    student_pref_section_dict = {}  # Key: student_id, Value: list of section_ids
    multisection_student_dict = {}  # Key: class_id, Value: list of student_ids who could be shuffled around in multisection classes

    for index, row in df_student_preferences.iterrows():
        if index == 0:
            continue
        student_id = row['StudentID']
        student_pref_section_dict[student_id] = []

        for i in range(2, 10, 1):
            class_id = row.iloc[i]
            if class_id == '' or pd.isna(class_id):
                continue
            # Find the class ID for this class ID
            section_list = class_section_dict.get(class_id, None)
            if section_list is None:
                # Print a warning if the class is not found in multisection_dict
                print(f"Warning: Class {class_id} not found in class_section_dict. Something is wrong.")
                continue
            elif len(section_list) == 1:
                # This is a single section class, so we can directly assign the section ID
                section_id = section_list[0]  # Assuming section_list has only one section ID
                student_pref_section_dict[student_id].append(section_id)
            else:
                # This is a multisection class.
                if class_id not in multisection_student_dict:
                    multisection_student_dict[class_id] = []
                multisection_student_dict[class_id].append(student_id)
    
    return student_pref_section_dict, multisection_student_dict

def handle_multisection_classes(student_pref_section_dict, multisection_student_dict, multi_section_dict):

    for class_id, student_ids in multisection_student_dict.items():
        if class_id == None:
            print("Warning: class_id is None. Skipping this class.")
            continue
        # Get the section IDs for this class_id
        section_list = multi_section_dict.get(class_id, None)
        if section_list is None or len(section_list)==1:
            print(f"Warning: Class {class_id} not found in multisection_dict. Something is wrong.")
            continue
        
        temp_dict = {}  # Key: section_id, Value: list of student_ids
        for section_id in section_list:
                temp_dict[section_id] = []
        balanced_sections = balance_sections_random(temp_dict, student_ids)
        # balanced_sections = balance_sections_according_to_student_pref(temp_dict, student_ids, df_student_preferences, professor_ID_dict)

        # Now, update the student_pref_section_dict with the balanced sections
        for section_id, students in balanced_sections.items():
            for student in students:
                if student not in student_pref_section_dict:
                    student_pref_section_dict[student] = []
                student_pref_section_dict[student].append(section_id)
        
    return student_pref_section_dict

def balance_sections_random(temp_dict, student_ids):
    # This function takes a dictionary of section IDs and their student IDs (some are empty, and some are filled), and a list of student IDs
    # It will balance the sections by randomly assigning students to sections so that the number of students in each section is as even as possible
    # But remember that some are already filled, so we need to take that into account
    # Returns a dictionary with section IDs as keys and lists of student IDs as values

    random.shuffle(student_ids)  # Shuffle the student IDs to randomize the assignment
    section_ids = list(temp_dict.keys())
    num_sections = len(section_ids)
    num_students = len(student_ids) + sum(len(temp_dict[section_id]) for section_id in section_ids)
    if num_sections == 0:
        return {}
    # Calculate the target number of students per section
    target_students_per_section = num_students // num_sections
    if num_students % num_sections != 0:
        target_students_per_section += 1  # If not evenly divisible, round up

    # Calculate the number of students that need to be added to each section
    for section_id in section_ids:
        students_to_add = target_students_per_section - len(temp_dict[section_id])
        for _ in range(students_to_add):
            if student_ids:
                student_id = student_ids.pop(0)
                temp_dict[section_id].append(student_id)
    return temp_dict

def initialize_enrollment_first(section_dict, time_block_dict, time_block_overlap_dict, room_dict, prof_pref_dict):
    time_to_sec, time_to_room = {}, {}
    sec_to_time, sec_to_room = {}, {}

    enrolled, non_enrolled = _enrolled_and_non_enrolled(section_dict, prof_pref_dict)

    # -------- 1) Assign enrolled sections (your existing logic, unchanged in spirit) --------
    all_rooms = list(room_dict.keys())

    def room_is_free_for_time(room_id, time_block):
        occupied_rooms = set(time_to_room.get(time_block, set()))
        for t in time_block_overlap_dict.get(time_block, []):
            occupied_rooms |= set(time_to_room.get(t, set()))
        return room_id not in occupied_rooms

    for section_id in enrolled:
        meta = prof_pref_dict[section_id]
        preferred_time_blocks_raw = meta.get('preferred_time_blocks') or []
        impossible_time_blocks = set(meta.get('impossible_time_blocks') or [])
        time_block_type = meta['time_block_type']
        preferred_rooms = meta.get('preferred_room_ids') or []
        required_feature = meta.get('required_room_feature', '')

        preferred_time_blocks = [tb for tb in preferred_time_blocks_raw if tb not in impossible_time_blocks]
        available_but_not_preferred_time_blocks = [
            tb for tb in time_block_dict[time_block_type] 
            if tb not in preferred_time_blocks and tb not in impossible_time_blocks
        ]

        possible_time_room_pairs = []

        # 1) preferred rooms at preferred times
        for room_id in preferred_rooms:
            if not _room_meets_required_feature(room_id, required_feature, room_dict):
                continue
            for time_block in preferred_time_blocks:
                if room_is_free_for_time(room_id, time_block) and \
                   _prof_is_free_for_time(section_id, time_block, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                    possible_time_room_pairs.append((time_block, room_id))

        # 2) any feature-ok room at preferred times
        if not possible_time_room_pairs:
            for room_id in all_rooms:
                if not _room_meets_required_feature(room_id, required_feature, room_dict):
                    continue
                for time_block in preferred_time_blocks:
                    if room_is_free_for_time(room_id, time_block) and \
                       _prof_is_free_for_time(section_id, time_block, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                        possible_time_room_pairs.append((time_block, room_id))

        # 3) any feature-ok room at allowed-but-not-preferred times
        if not possible_time_room_pairs:
            for room_id in all_rooms:
                if not _room_meets_required_feature(room_id, required_feature, room_dict):
                    continue
                for time_block in available_but_not_preferred_time_blocks:
                    if room_is_free_for_time(room_id, time_block) and \
                       _prof_is_free_for_time(section_id, time_block, time_to_sec, time_block_overlap_dict, prof_pref_dict):
                        possible_time_room_pairs.append((time_block, room_id))

        if not possible_time_room_pairs:
            # leave unassigned; validator will show it
            # print(f"Warning: No available time/room for enrolled section {section_id}")
            print(f"Warning: No available time/room for enrolled section {section_id}, leaving unassigned")
            continue

        # choose randomly among feasible options to keep diversity for hill-climb
        tb, rm = random.choice(possible_time_room_pairs)
        time_to_sec.setdefault(tb, set()).add(section_id)
        time_to_room.setdefault(tb, set()).add(rm)
        sec_to_time[section_id] = tb
        sec_to_room[section_id] = rm

    # -------- 2) Assign non-enrolled sections (prefer preferred slots) --------
    for section_id in non_enrolled:
        placed = _choose_slots_for(section_id, time_block_dict, time_block_overlap_dict,
                                   room_dict, prof_pref_dict, time_to_sec, time_to_room,
                                   prefer_strict=True)
        if placed:
            tb, rm = placed
            time_to_sec.setdefault(tb, set()).add(section_id)
            time_to_room.setdefault(tb, set()).add(rm)
            sec_to_time[section_id] = tb
            sec_to_room[section_id] = rm
        # else: keep unassigned; not fatal since they don’t affect student conflicts

    return time_to_sec, time_to_room, sec_to_time, sec_to_room

#########################################################

def prof_free_at_for_move(section_id, new_tb, sec_to_time, prof_pref_dict, time_block_overlap_dict):
    """Like prof_free_at, but ignore this section's current occupancy when it overlaps."""
    pid = prof_pref_dict[section_id].get('professor_id')
    if not pid:
        return True
    new_clos = set(time_block_overlap_dict.get(new_tb, ()))
    old_tb = sec_to_time.get(section_id)
    old_clos = set(time_block_overlap_dict.get(old_tb, ())) if old_tb else set()
    # Overlaps the professor has at new_clos
    overlaps = new_clos & prof_busy_blocks[pid]
    # If the only overlaps are from THIS section's current slot, allow it
    return len(overlaps - old_clos) == 0

def room_free_at_for_move(room_id, section_id, new_tb, sec_to_time, sec_to_room, time_block_overlap_dict):
    """Like room_free_at, but ignore this section's current occupancy if staying in the same room."""
    new_clos = set(time_block_overlap_dict.get(new_tb, ()))
    overlaps = new_clos & room_busy_blocks[room_id]
    cur_rm = sec_to_room.get(section_id)
    if room_id == cur_rm:
        old_tb = sec_to_time.get(section_id)
        old_clos = set(time_block_overlap_dict.get(old_tb, ())) if old_tb else set()
        overlaps = overlaps - old_clos
    return len(overlaps) == 0

def apply_change_fast(section_id, new_time_block, new_room,
                      time_to_sec, time_to_room, sec_to_time, sec_to_room,
                      prof_pref_dict, time_block_overlap_dict):
    
    if section_id not in sec_to_time or section_id not in sec_to_room:
        return  # not currently assigned; nothing to swap from

    old_tb = sec_to_time[section_id]
    old_rm = sec_to_room[section_id]

    # Unmark old occupancy first
    _unmark_occupancy_for_assignment(section_id, old_tb, old_rm, prof_pref_dict, time_block_overlap_dict)

    # Update time_to_sec
    time_to_sec[old_tb].remove(section_id)
    if not time_to_sec[old_tb]:
        del time_to_sec[old_tb]
    time_to_sec.setdefault(new_time_block, set()).add(section_id)

    # Update time_to_room
    time_to_room[old_tb].remove(old_rm)
    if not time_to_room[old_tb]:
        del time_to_room[old_tb]
    time_to_room.setdefault(new_time_block, set()).add(new_room)

    # Update sec_to_*
    sec_to_time[section_id] = new_time_block
    sec_to_room[section_id] = new_room

    # Mark new occupancy
    _mark_occupancy_for_assignment(section_id, new_time_block, new_room, prof_pref_dict, time_block_overlap_dict)

def hill_climbing_optimization_fast(
    section_dict,
    time_block_overlap_dict,
    prof_pref_dict,
    adjacency_matrix,
    adjacency_matrix_order_map,
    time_to_sec,
    time_to_room,
    sec_to_time,
    sec_to_room,
    allowed_times_for_section,
    feature_ok_rooms_for_section,
    preferred_times_set,
    time_penalty=5,
    room_penalty=3,
    first_improvement=True,
):
    """
    Returns:
        moves_made (int): how many improving moves were applied
        feasible_candidates (int): how many (time,room) candidates were actually feasible,
                                   regardless of improvement; if this is 0 the schedule is 'stuck'
    """
    # Only consider sections that are currently assigned (avoid KeyError)
    assigned_sections = [s for s in section_dict.keys() if s in sec_to_time and s in sec_to_room]
    sections = random.sample(assigned_sections, len(assigned_sections))

    section_to_idx = {sec: i for i, sec in enumerate(adjacency_matrix_order_map)}

    def _penalty(sec, tb, rm):
        pen = 0
        if tb not in preferred_times_set.get(sec, ()):
            pen += time_penalty
        if rm not in (prof_pref_dict[sec].get('preferred_room_ids') or []):
            pen += room_penalty
        return pen

    moves_made = 0
    feasible_candidates = 0

    for sec in sections:
        # defensive: skip if not fully assigned
        if sec not in sec_to_time or sec not in sec_to_room:
            continue

        cur_tb = sec_to_time[sec]
        cur_rm = sec_to_room[sec]
        i = section_to_idx[sec]

        # compute current local conflict
        vis_cur = set()
        for u in time_block_overlap_dict.get(cur_tb, ()):
            vis_cur |= set(time_to_sec.get(u, ()))
        vis_cur.discard(sec)

        cur_conflict = 0
        for o in vis_cur:
            j = section_to_idx[o]
            cur_conflict += adjacency_matrix[i][j]
        cur_cost = cur_conflict + _penalty(sec, cur_tb, cur_rm)

        best_delta = 0
        best_move = None

        cand_times = allowed_times_for_section.get(sec, [])
        pref_set = preferred_times_set.get(sec, set())
        cand_times_ordered = [t for t in cand_times if t in pref_set and t != cur_tb] + \
                             [t for t in cand_times if t not in pref_set and t != cur_tb]

        for new_tb in cand_times_ordered:
            if not prof_free_at_for_move(sec, new_tb, sec_to_time, prof_pref_dict, time_block_overlap_dict):
                continue

            vis_new = set()
            for u in time_block_overlap_dict.get(new_tb, ()):
                vis_new |= set(time_to_sec.get(u, ()))
            vis_new.discard(sec)

            new_conflict = 0
            for o in vis_new:
                j = section_to_idx[o]
                new_conflict += adjacency_matrix[i][j]

            had_feasible_for_this_time = False
            for new_rm in feature_ok_rooms_for_section.get(sec, []):
                if not room_free_at_for_move(new_rm, sec, new_tb, sec_to_time, sec_to_room, time_block_overlap_dict):
                    continue

                # At least one feasible (time,room) candidate exists
                feasible_candidates += 1
                had_feasible_for_this_time = True

                new_cost = new_conflict + _penalty(sec, new_tb, new_rm)
                delta = new_cost - cur_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = (new_tb, new_rm, new_cost)
                    if first_improvement:
                        break
            if first_improvement and best_move is not None:
                break

        if best_move is not None:
            new_tb, new_rm, _ = best_move
            apply_change_fast(
                sec, new_tb, new_rm,
                time_to_sec, time_to_room, sec_to_time, sec_to_room,
                prof_pref_dict, time_block_overlap_dict
            )
            moves_made += 1

    return moves_made, feasible_candidates

#########################################################

def compute_total_cost(adjacency_matrix, adjacency_matrix_order_map,
                       time_block_overlap_dict, sec_to_time, sec_to_room, time_to_sec, prof_pref_dict, time_penalty=5, room_penalty=3):
    """
    Sum conflict costs once per unordered pair (i<j), plus per-section penalties once.
    """
    total_conflict_cost = 0
    index_of = {sec_id: idx for idx, sec_id in enumerate(adjacency_matrix_order_map)}

    # Sum conflicts once
    for sec_i, tb_i in sec_to_time.items():
        i = index_of[sec_i]
        times_i = set(time_block_overlap_dict.get(tb_i, []))

        overlapping_sections = set()
        for t in times_i:
            overlapping_sections |= set(time_to_sec.get(t, set()))

        for sec_j in overlapping_sections:
            if sec_j == sec_i:
                continue
            j = index_of[sec_j]
            if i < j:  # count each unordered pair once
                total_conflict_cost += adjacency_matrix[i][j]

    total_penalty = 0
    for sec_id in sec_to_time.keys():
        tb = sec_to_time[sec_id]
        rm = sec_to_room[sec_id]
        if tb and rm:
            pref_times = prof_pref_dict[sec_id].get('preferred_time_blocks') or []
            pref_rooms = prof_pref_dict[sec_id].get('preferred_room_ids') or []
            pen = (0 if tb in pref_times else time_penalty) + (0 if rm in pref_rooms else room_penalty)
            total_penalty += pen
    return total_conflict_cost + total_penalty

def repair_non_enrolled_after_climb(section_dict, time_block_dict, time_block_overlap_dict,
                                    room_dict, prof_pref_dict, time_to_sec, time_to_room,
                                    sec_to_time, sec_to_room):
    """
    After hill-climb moves (which operated on enrolled sections),
    re-validate and (re)place non-enrolled sections using current availability.

    Keeps prof_busy_blocks / room_busy_blocks in sync:
      - unmarks old occupancy before removal
      - marks new occupancy after placement
    """

    # Helpers that check the *current* assignment is valid while IGNORING the section itself.
    # We use time_to_sec / sec_to_room here (not the busy sets), because busy sets include 'self'.
    def _room_ok_ignoring_self(sec_id, tb, rm):
        for u in time_block_overlap_dict.get(tb, ()):
            for other in time_to_sec.get(u, ()):
                if other != sec_id and sec_to_room.get(other) == rm:
                    return False
        return True

    def _prof_ok_ignoring_self(sec_id, tb):
        pid = prof_pref_dict[sec_id].get('professor_id')
        if not pid:
            return True
        for u in time_block_overlap_dict.get(tb, ()):
            for other in time_to_sec.get(u, ()):
                if other != sec_id and prof_pref_dict[other].get('professor_id') == pid:
                    return False
        return True

    # Non-enrolled = defined in prof_pref_dict but not present in 'section_dict' (no students driving conflicts)
    enrolled = set(section_dict.keys())
    all_secs = set(prof_pref_dict.keys())
    non_enrolled = list(all_secs - enrolled)

    for s in non_enrolled:
        tb = sec_to_time.get(s)
        rm = sec_to_room.get(s)
        req_feat = prof_pref_dict[s].get('required_room_feature', '')

        # If already assigned, check that assignment is still valid (ignoring self)
        ok = False
        if tb and rm:
            req_ok  = _room_meets_required_feature(rm, req_feat, room_dict)
            room_ok = _room_ok_ignoring_self(s, tb, rm)
            prof_ok = _prof_ok_ignoring_self(s, tb)
            ok = req_ok and room_ok and prof_ok

        if ok:
            # Still valid; keep it.
            continue

        # ---- Remove stale assignment (if any) and unmark occupancy trackers
        if tb and rm:
            _unmark_occupancy_for_assignment(s, tb, rm, prof_pref_dict, time_block_overlap_dict)

            if s in time_to_sec.get(tb, set()):
                time_to_sec[tb].discard(s)
                if not time_to_sec[tb]:
                    time_to_sec.pop(tb, None)

            # Remove the room usage for this block
            if tb in time_to_room:
                time_to_room[tb].discard(rm)
                if not time_to_room[tb]:
                    time_to_room.pop(tb, None)

        sec_to_time.pop(s, None)
        sec_to_room.pop(s, None)

        # ---- Try to (re)place the section using your existing slot picker
        placed = _choose_slots_for(
            s, time_block_dict, time_block_overlap_dict,
            room_dict, prof_pref_dict, time_to_sec, time_to_room,
            prefer_strict=True
        )

        if placed:
            new_tb, new_rm = placed
            time_to_sec.setdefault(new_tb, set()).add(s)
            time_to_room.setdefault(new_tb, set()).add(new_rm)
            sec_to_time[s] = new_tb
            sec_to_room[s] = new_rm

            # Mark new occupancy in the fast trackers
            _mark_occupancy_for_assignment(s, new_tb, new_rm, prof_pref_dict, time_block_overlap_dict)
        # else: stays unassigned (intentionally); no penalty entry for non-enrolled

#########################################################

def validate_schedule(sec_to_time, sec_to_room, time_block_overlap_dict,
                      room_dict, prof_pref_dict,
                      adjacency_matrix=None, adjacency_matrix_order_map=None,
                      time_to_sec=None, time_penalty=10, room_penalty=5):
    """
    Returns a dict with:
      - unassigned_sections, missing_time, missing_room
      - room_conflicts: (room_id, sec_a, tb_a, sec_b, tb_b)
      - professor_conflicts: (prof_id, sec_a, tb_a, sec_b, tb_b)
      - impossible_time_assignments: (section_id, tb, impossible_list)
      - missing_required_features: (section_id, room_id, missing_features_csv)
      - non_preferred_time_sections: [section_id, ...]
      - non_preferred_room_sections: [section_id, ...]
      - conflict_cost_total: int        # counted once per unordered pair (i<j)
      - penalty_cost_total: int         # penalties for sections that appear in the adjacency matrix only
      - total_cost: int                 # conflict_cost_total + penalty_cost_total
    """
    report = {
        "unassigned_sections": [],
        "missing_time": [],
        "missing_room": [],
        "room_conflicts": [],
        "professor_conflicts": [],
        "impossible_time_assignments": [],
        "missing_required_features": [],
        "non_preferred_time_sections": [],
        "non_preferred_room_sections": [],
        "conflict_cost_total": 0,
        "penalty_cost_total": 0,
        "total_cost": 0,
    }

    def blocks_overlap(tb1, tb2):
        return (tb1 == tb2) or (tb2 in (time_block_overlap_dict.get(tb1, []) or []))

    section_prof = {sid: prof_pref_dict[sid].get('professor_id') for sid in prof_pref_dict.keys()}
    section_reqf = {sid: prof_pref_dict[sid].get('required_room_feature', '') for sid in prof_pref_dict.keys()}
    section_pref_times = {sid: prof_pref_dict[sid].get('preferred_time_blocks') or [] for sid in prof_pref_dict.keys()}
    section_pref_rooms = {sid: prof_pref_dict[sid].get('preferred_room_ids') or [] for sid in prof_pref_dict.keys()}

    # --- 0) Unassigned + track non-preferred time/room (for all assigned sections)
    for section_id in prof_pref_dict.keys():
        tb = sec_to_time.get(section_id)
        rm = sec_to_room.get(section_id)
        if not tb and not rm:
            report["unassigned_sections"].append(section_id)
            continue
        if not tb:
            report["missing_time"].append(section_id)
        if not rm:
            report["missing_room"].append(section_id)

        # non-preferred flags (only if assigned)
        if tb and tb not in section_pref_times[section_id]:
            report["non_preferred_time_sections"].append(section_id)
        if rm and rm not in section_pref_rooms[section_id]:
            report["non_preferred_room_sections"].append(section_id)

    # --- 1) Room conflicts
    room_usage = {}
    for sid, rm in sec_to_room.items():
        tb = sec_to_time.get(sid)
        if not tb or not rm:
            continue
        room_usage.setdefault(rm, []).append((sid, tb))

    for rm, lst in room_usage.items():
        for i in range(len(lst)):
            sid_i, tb_i = lst[i]
            for j in range(i+1, len(lst)):
                sid_j, tb_j = lst[j]
                if blocks_overlap(tb_i, tb_j):
                    report["room_conflicts"].append((rm, sid_i, tb_i, sid_j, tb_j))

    # --- 2) Professor conflicts
    prof_usage = {}
    for sid, tb in sec_to_time.items():
        prof_id = section_prof.get(sid)
        if not tb or not prof_id:
            continue
        prof_usage.setdefault(prof_id, []).append((sid, tb))

    for pid, lst in prof_usage.items():
        for i in range(len(lst)):
            sid_i, tb_i = lst[i]
            for j in range(i+1, len(lst)):
                sid_j, tb_j = lst[j]
                if blocks_overlap(tb_i, tb_j):
                    report["professor_conflicts"].append((pid, sid_i, tb_i, sid_j, tb_j))

    # --- 3) Impossible time assignments
    for sid, tb in sec_to_time.items():
        if not tb:
            continue
        impossible = set(prof_pref_dict[sid].get('impossible_time_blocks') or [])
        if tb in impossible:
            report["impossible_time_assignments"].append((sid, tb, sorted(impossible)))

    # --- 4) Required features
    for sid, rm in sec_to_room.items():
        if not rm:
            continue
        req = _normalize_features(section_reqf.get(sid, ''))
        if not req:
            continue
        feats = _normalize_features(room_dict.get(rm, {}).get('features', ''))
        missing = req - feats
        if missing:
            report["missing_required_features"].append((sid, rm, ",".join(sorted(missing))))

    # --- 5) Optional cost breakdown (requires adjacency & time_to_sec to be passed)
    if adjacency_matrix is not None and adjacency_matrix_order_map is not None and time_to_sec is not None:
        index_of = {sec_id: idx for idx, sec_id in enumerate(adjacency_matrix_order_map)}

        # Conflicts counted once per unordered pair (i<j)
        conflict_total = 0
        for sec_i, tb_i in sec_to_time.items():
            if sec_i not in index_of or not tb_i:
                continue
            i = index_of[sec_i]
            times_i = set(time_block_overlap_dict.get(tb_i, []))
            overlapping_sections = set()
            for t in times_i:
                overlapping_sections |= set(time_to_sec.get(t, set()))
            for sec_j in overlapping_sections:
                if sec_j == sec_i or sec_j not in index_of:
                    continue
                j = index_of[sec_j]
                if i < j:
                    conflict_total += adjacency_matrix[i][j]

        # Penalties only for sections that participate in adjacency (enrolled)
        penalty_total = 0
        for sid, tb in sec_to_time.items():
            if sid not in index_of or not tb:
                continue
            if tb not in section_pref_times[sid]:
                penalty_total += time_penalty
            rm = sec_to_room.get(sid)
            if rm and rm not in section_pref_rooms[sid]:
                penalty_total += room_penalty

        report["conflict_cost_total"] = conflict_total
        report["penalty_cost_total"] = penalty_total
        report["total_cost"] = conflict_total + penalty_total

    return report

#########################################################

def run_optimizer(input_dir: Path, output_dir: Path, seed: int | None = None):

    if seed is not None:
        random.seed(seed)

    """
    Runs the scheduler optimization using CSVs in input_dir and saves results to output_dir.
    Returns: Path to best_assignment.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs
    df_timeblocks     = pd.read_csv(input_dir / "time_blocks.csv")
    df_timeblock_dict = pd.read_csv(input_dir / "time_block_dict.csv")
    df_rooms          = pd.read_csv(input_dir / "rooms.csv")
    df_courses        = pd.read_csv(input_dir / "course_offered.csv")
    df_student_prefs  = pd.read_csv(input_dir / "student_preference.csv")
    df_prof_prefs     = pd.read_csv(input_dir / "professor_preference.csv")

    # ---- Precompute core dicts
    time_block_overlap_map, block_metadata, time_block_overlap_dict = compute_time_block_overlap_map(df_timeblocks)
    time_block_dict  = read_time_block_dict(df_timeblock_dict)   # {time_block_type: [block_ids]}
    room_dict        = build_room_dict(df_rooms)                 # {room_id: {...}}
    prof_pref_dict   = build_prof_pref_dict(df_prof_prefs)       # {section_id: {...}}

    # Features as sets
    for rm in room_dict.values():
        rm['features_set'] = _normalize_features(rm.get('features', ''))

    # Feasible rooms / allowed times / preferred sets
    feature_ok_rooms_for_section = {}
    allowed_times_for_section    = {}
    preferred_times_set          = {}
    preferred_rooms_set          = {}

    for sec, meta in prof_pref_dict.items():
        req = _normalize_features(meta.get('required_room_feature', ''))
        ok_rooms = [r for r, info in room_dict.items() if req.issubset(info['features_set'])]

        pref_rooms  = [r for r in (meta.get('preferred_room_ids') or []) if r in ok_rooms]
        other_rooms = [r for r in ok_rooms if r not in pref_rooms]
        feature_ok_rooms_for_section[sec] = pref_rooms + other_rooms

        ttype      = meta['time_block_type']
        impossible = set(meta.get('impossible_time_blocks') or [])
        allowed    = [tb for tb in (time_block_dict.get(ttype, []) or []) if tb not in impossible]
        allowed_times_for_section[sec] = allowed

        preferred_times_set[sec] = set(meta.get('preferred_time_blocks') or [])
        preferred_rooms_set[sec] = set(meta.get('preferred_room_ids') or [])

    # Build class→sections and initial student dicts
    class_section_dict                = create_class_section_dict(df_courses)
    multi_section_dict                = create_multi_section_dict(df_courses)
    temp_student_pref_section_dict, multisection_student_dict = update_student_preferences_for_single_section_classes(
        df_student_prefs, class_section_dict
    )

    # ---- Optimization parameters (tune as needed or expose as API params)
    N_DISTRIBUTIONS   = 10
    num_iterations    = 5
    time_penalty      = 1
    room_penalty      = 0
    good_enough_cost  = 0

    # Clear global trackers for a clean run
    prof_busy_blocks.clear()
    room_busy_blocks.clear()

    best_total_cost = float('inf')
    best_assignment = None

    for dist_idx in tqdm(range(N_DISTRIBUTIONS), desc="Evaluating student distributions",
                     disable=not sys.stdout.isatty()):
        base_pref = copy.deepcopy(temp_student_pref_section_dict)

        # Randomly balance multisection classes into sections
        student_pref_section_dict = handle_multisection_classes(
            base_pref, multisection_student_dict, multi_section_dict
        )

        # Build section -> students map
        section_dict = student_pref_to_section_dict(student_pref_section_dict, prof_pref_dict)

        # Build adjacency for sections with enrollment
        adjacency_matrix, adjacency_matrix_order_map = build_adjacency_matrix(section_dict)

        # Initial placement
        time_to_sec, time_to_room, sec_to_time, sec_to_room = initialize_enrollment_first(
            section_dict, time_block_dict, time_block_overlap_dict, room_dict, prof_pref_dict
        )

        # Seed occupancy trackers from the initial assignment
        prof_busy_blocks.clear()
        room_busy_blocks.clear()
        for sec, tb in sec_to_time.items():
            rm = sec_to_room.get(sec)
            if tb and rm:
                _mark_occupancy_for_assignment(sec, tb, rm, prof_pref_dict, time_block_overlap_dict)

        # Cost before optimization
        total_conflict_cost = compute_total_cost(
            adjacency_matrix, adjacency_matrix_order_map,
            time_block_overlap_dict, sec_to_time, sec_to_room,
            time_to_sec, prof_pref_dict, time_penalty, room_penalty
        )

        restart_distribution = False

        # Hill-climbing loop
        for _ in range(num_iterations):
            snap = _snapshot_state(
                time_to_sec, time_to_room, sec_to_time, sec_to_room,
                prof_busy_blocks, room_busy_blocks
            )
            try:
                moves_made, feasible_candidates = hill_climbing_optimization_fast(
                    section_dict, time_block_overlap_dict, 
                    prof_pref_dict, adjacency_matrix, adjacency_matrix_order_map,
                    time_to_sec, time_to_room, sec_to_time, sec_to_room, 
                    allowed_times_for_section, feature_ok_rooms_for_section, preferred_times_set,
                    time_penalty, room_penalty, first_improvement=True
                )
            except Exception:
                _restore_state(snap, time_to_sec, time_to_room, sec_to_time, sec_to_room,
                               prof_busy_blocks, room_busy_blocks)
                _rebuild_occupancy_from_assignments(sec_to_time, sec_to_room, prof_pref_dict, time_block_overlap_dict)
                restart_distribution = True
                break

            if feasible_candidates == 0:
                _restore_state(snap, time_to_sec, time_to_room, sec_to_time, sec_to_room,
                               prof_busy_blocks, room_busy_blocks)
                _rebuild_occupancy_from_assignments(sec_to_time, sec_to_room, prof_pref_dict, time_block_overlap_dict)
                restart_distribution = True
                break

            if moves_made == 0:
                break

        if restart_distribution:
            continue

        # Place non-enrolled after climbing
        repair_non_enrolled_after_climb(
            section_dict, time_block_dict, time_block_overlap_dict,
            room_dict, prof_pref_dict, time_to_sec, time_to_room,
            sec_to_time, sec_to_room
        )

        # Final cost for this distribution
        total_conflict_cost = compute_total_cost(
            adjacency_matrix, adjacency_matrix_order_map,
            time_block_overlap_dict, sec_to_time, sec_to_room,
            time_to_sec, prof_pref_dict, time_penalty, room_penalty
        )

        if total_conflict_cost < best_total_cost:
            best_total_cost = total_conflict_cost
            best_assignment = {
                'sec_to_time': dict(sec_to_time),
                'sec_to_room': dict(sec_to_room),
                'time_to_sec': {k: set(v) for k, v in time_to_sec.items()},
                'time_to_room': {k: set(v) for k, v in time_to_room.items()},
                'adjacency_matrix': [row[:] for row in adjacency_matrix],
                'adjacency_matrix_order_map': adjacency_matrix_order_map[:],
                'total_cost': total_conflict_cost,
            }

        if total_conflict_cost <= good_enough_cost:
            break

    # ---- Persist best assignment & report
    if not best_assignment:
        # Edge case: nothing assigned; still return a minimal JSON
        result_path = output_dir / "best_assignment.json"
        with open(result_path, "w") as f:
            json.dump({"error": "No feasible assignment found"}, f, indent=2)
        return result_path

    best_sec_to_time = best_assignment['sec_to_time']
    best_sec_to_room = best_assignment['sec_to_room']
    best_time_to_sec_sets = best_assignment['time_to_sec']
    best_time_to_room_sets = best_assignment['time_to_room']
    best_adj   = best_assignment['adjacency_matrix']
    best_order = best_assignment['adjacency_matrix_order_map']

    final_report = validate_schedule(
        best_sec_to_time, best_sec_to_room, time_block_overlap_dict,
        room_dict, prof_pref_dict,
        adjacency_matrix=best_adj,
        adjacency_matrix_order_map=best_order,
        time_to_sec=best_time_to_sec_sets,
        time_penalty=time_penalty, room_penalty=room_penalty
    )

    # JSON (full)
    result_path = output_dir / "best_assignment.json"
    with open(result_path, "w") as f:
        json.dump({
            'sec_to_time': best_sec_to_time,
            'sec_to_room': best_sec_to_room,
            'time_to_sec': {k: sorted(list(v)) for k, v in best_time_to_sec_sets.items()},
            'time_to_room': {k: sorted(list(v)) for k, v in best_time_to_room_sets.items()},
            'adjacency_matrix': best_adj,
            'adjacency_matrix_order_map': best_order,
            'total_cost': best_assignment['total_cost'],
            'report': final_report,
        }, f, indent=2)

    # CSVs (simple)
    pd.DataFrame([
        {"Section ID": s, "Time Block": tb}
        for s, tb in best_sec_to_time.items()
    ]).to_csv(output_dir / "schedule_time.csv", index=False)

    pd.DataFrame([
        {"Section ID": s, "Room ID": rm}
        for s, rm in best_sec_to_room.items()
    ]).to_csv(output_dir / "schedule_room.csv", index=False)

    return result_path
