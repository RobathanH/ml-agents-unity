using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A SensorComponent that builds a variable-length entity tensor suitable for EGNN encoders.
    /// It allows selecting multiple root GameObjects (e.g., self, opponent, ground) and aggregates
    /// per-child Rigidbody nodes with positions and auxiliary attributes. The resulting observation
    /// is provided via a BufferSensor of fixed per-entity size.
    /// </summary>
    [AddComponentMenu("ML Agents/EGNN Sensor", (int)MenuGroup.Sensors)]
    public class EGNNSensorComponent : SensorComponent
    {
        [Serializable]
        public class RootGroup
        {
            public GameObject Root;
            public string Type; // e.g., "self", "opponent", "ground"
            public bool IncludeChildren = true;
            public string RootSubType = "";
            public List<ChildOverride> Children = new List<ChildOverride>();
        }

        [Serializable]
        public class ChildOverride
        {
            public Transform Child;
            public bool Include = true;
            public string SubType = "";
        }

        /// <summary>
        /// Name of the generated BufferSensor.
        /// </summary>
        [SerializeField]
        private string m_SensorName = "EGNNSensor";

        /// <summary>
        /// Maximum number of entities to include (padding beyond this will be zeros).
        /// </summary>
        [SerializeField]
        private int m_MaxEntities = 128;

        // Frozen category plan. Built once in CreateSensors and never recomputed:
        // the BufferSensor row width is allocated from these caps, so the pass that
        // sizes a row and the pass that writes it must agree by construction.
        // They previously did not. ComputeObservableSize counted statically
        // configured non-empty SubType strings (including entries with Include=0),
        // while Update counted labels discovered at runtime (falling back to
        // GameObject names when SubType was blank) and then overwrote the caps.
        // Both rules happened to yield 4 for CrawlerSumo -- {body, leg, foreleg,
        // Sweatband} vs {body, leg, foreleg, Cylinder} -- so rows were the correct
        // width by coincidence. Dropping the excluded Sweatband override, or naming
        // the ground child's SubType, would have changed the written row length
        // after the sensor had already been allocated.
        private int m_TypeCap = 1;
        private int m_SubTypeCap = 1;
        private readonly Dictionary<string, int> m_TypeIndex = new Dictionary<string, int>();
        private readonly Dictionary<string, int> m_SubTypeIndex = new Dictionary<string, int>();
        private bool m_PlanBuilt;
        private bool m_WarnedUnknownLabel;

        /// <summary>
        /// Whether to include per-entity orientation (quaternion) as attributes.
        /// </summary>
        [SerializeField]
        private bool m_IncludeRotation = false;

        /// <summary>
        /// Whether to include per-entity linear velocity as attributes (model or world space).
        /// </summary>
        [SerializeField]
        private bool m_IncludeLinearVelocity = true;

        /// <summary>
        /// Whether to include per-entity angular velocity as attributes.
        /// </summary>
        [SerializeField]
        private bool m_IncludeAngularVelocity = false;

        /// <summary>
        /// Optional virtual root to stabilize relative coordinates (model space reference).
        /// If set, positions and velocities are expressed in this frame.
        /// </summary>
        [SerializeField]
        private GameObject m_VirtualRoot;

        /// <summary>
        /// Configurable root groups for entity extraction.
        /// </summary>
        [SerializeField]
        private List<RootGroup> m_RootGroups = new List<RootGroup>();

        private BufferSensor m_Sensor;

        // Per-entity feature size: 3 pos + optional attrs + one-hot(type) + one-hot(subtype)
        private int ComputeObservableSize()
        {
            int size = 3; // position
            if (m_IncludeRotation) size += 4;
            if (m_IncludeLinearVelocity) size += 3;
            if (m_IncludeAngularVelocity) size += 3;
            if (!m_PlanBuilt)
            {
                BuildCategoryPlan();
            }
            size += Mathf.Max(1, m_TypeCap);
            size += Mathf.Max(1, m_SubTypeCap);
            return size;
        }

        public override ISensor[] CreateSensors()
        {
            // Freeze the label -> index maps and the resulting caps
            BuildCategoryPlan();
            // Infer capacity from roots and included children
            m_MaxEntities = InferMaxEntities();
            int obsSize = ComputeObservableSize();
            m_Sensor = new BufferSensor(m_MaxEntities, obsSize, m_SensorName);
            return new ISensor[] { m_Sensor };
        }

        private void Update()
        {
            if (m_Sensor == null)
            {
                return;
            }
            m_Sensor.Reset();

            // Gather transforms
            List<(Vector3 pos, Quaternion rot, Vector3 linVel, Vector3 angVel, int typeId, int subTypeId)> entities = new List<(Vector3, Quaternion, Vector3, Vector3, int, int)>();

            foreach (var group in m_RootGroups)
            {
                if (group.Root == null)
                {
                    continue;
                }
                // Root-level type id (resolved against the frozen plan)
                int tidRaw = LookupCategory(m_TypeIndex, TypeLabel(group), "type");

                // Root entity
                var rootRb = group.Root.GetComponent<Rigidbody>();
                var rootCol = group.Root.GetComponent<Collider>();
                if (rootRb != null || rootCol != null)
                {
                    int sidRaw = LookupCategory(m_SubTypeIndex, RootSubTypeLabel(group), "subtype");
                    if (rootRb != null)
                    {
                        entities.Add((rootRb.position, rootRb.rotation, rootRb.linearVelocity, rootRb.angularVelocity, tidRaw, sidRaw));
                    }
                    else
                    {
                        var tr = group.Root.transform;
                        entities.Add((tr.position, tr.rotation, Vector3.zero, Vector3.zero, tidRaw, sidRaw));
                    }
                }

                if (entities.Count >= m_MaxEntities) break;

                // Children selection
                if (group.IncludeChildren)
                {
                    // Ensure overrides are synchronized
                    SyncChildrenForGroup(group);
                    foreach (var child in group.Children)
                    {
                        if (!child.Include || child.Child == null) continue;
                        var childRb = child.Child.GetComponent<Rigidbody>();
                        var childCol = child.Child.GetComponent<Collider>();
                        if (childRb == null && childCol == null) continue;
                        int sidRaw2 = LookupCategory(m_SubTypeIndex, ChildSubTypeLabel(child), "subtype");
                        if (childRb != null)
                        {
                            entities.Add((childRb.position, childRb.rotation, childRb.linearVelocity, childRb.angularVelocity, tidRaw, sidRaw2));
                        }
                        else
                        {
                            var trc = child.Child.transform;
                            entities.Add((trc.position, trc.rotation, Vector3.zero, Vector3.zero, tidRaw, sidRaw2));
                        }
                        if (entities.Count >= m_MaxEntities) break;
                    }
                }
                if (entities.Count >= m_MaxEntities) break;
            }

            // Coordinate transform to virtual root if provided
            if (m_VirtualRoot != null)
            {
                var invRot = Quaternion.Inverse(m_VirtualRoot.transform.rotation);
                var origin = m_VirtualRoot.transform.position;
                for (int i = 0; i < entities.Count; i++)
                {
                    var e = entities[i];
                    var relPos = invRot * (e.pos - origin);
                    var relVel = invRot * e.linVel;
                    var relAng = invRot * e.angVel;
                    var relRot = invRot * e.rot;
                    entities[i] = (relPos, relRot, relVel, relAng, e.typeId, e.subTypeId);
                }
            }

            // Write to buffer sensor as fixed-size rows; remaining rows stay zero as
            // padding. Caps are NOT recomputed here -- they are frozen in
            // BuildCategoryPlan so the row width matches the allocated sensor.
            foreach (var e in entities)
            {
                var row = BuildRow(e.pos, e.rot, e.linVel, e.angVel, e.typeId, e.subTypeId);
                m_Sensor.AppendObservation(row);
            }
        }

        private float[] BuildRow(Vector3 pos, Quaternion rot, Vector3 linVel, Vector3 angVel, int typeId, int subTypeId)
        {
            var obs = new List<float>(ComputeObservableSize());
            // Position first (for EGNN positional dims)
            obs.Add(pos.x); obs.Add(pos.y); obs.Add(pos.z);
            if (m_IncludeRotation)
            {
                obs.Add(rot.x); obs.Add(rot.y); obs.Add(rot.z); obs.Add(rot.w);
            }
            if (m_IncludeLinearVelocity)
            {
                obs.Add(linVel.x); obs.Add(linVel.y); obs.Add(linVel.z);
            }
            if (m_IncludeAngularVelocity)
            {
                obs.Add(angVel.x); obs.Add(angVel.y); obs.Add(angVel.z);
            }
            // Type one-hot (fixed size)
            int typeSize = Math.Max(1, m_TypeCap);
            for (int i = 0; i < typeSize; i++)
            {
                obs.Add(i == Mathf.Clamp(typeId, 0, typeSize - 1) ? 1f : 0f);
            }
            // SubType one-hot (fixed size)
            int subSize = Math.Max(1, m_SubTypeCap);
            for (int i = 0; i < subSize; i++)
            {
                obs.Add(i == Mathf.Clamp(subTypeId, 0, subSize - 1) ? 1f : 0f);
            }
            return obs.ToArray();
        }

        internal void SyncChildrenForGroup(RootGroup group)
        {
            if (group == null || group.Root == null)
            {
                return;
            }
            var existing = new Dictionary<Transform, ChildOverride>();
            if (group.Children != null)
            {
                foreach (var c in group.Children)
                {
                    if (c != null && c.Child != null && !existing.ContainsKey(c.Child))
                    {
                        existing[c.Child] = c;
                    }
                }
            }
            var newList = new List<ChildOverride>();
            var seen = new HashSet<Transform>();
            // Gather all Rigidbodies under root
            var rbs = group.Root.GetComponentsInChildren<Rigidbody>();
            foreach (var rb in rbs)
            {
                var tr = rb.transform;
                if (tr == group.Root.transform)
                {
                    continue; // root handled separately
                }
                if (seen.Contains(tr))
                {
                    continue;
                }
                seen.Add(tr);
                if (existing.TryGetValue(tr, out var co))
                {
                    newList.Add(co);
                }
                else
                {
                    newList.Add(new ChildOverride { Child = tr, Include = true, SubType = tr.name });
                }
            }
            // Gather all Colliders under root (any collider type), avoid duplicates
            var cols = group.Root.GetComponentsInChildren<Collider>();
            foreach (var col in cols)
            {
                var tr = col.transform;
                if (tr == group.Root.transform)
                {
                    continue; // root handled separately
                }
                if (seen.Contains(tr))
                {
                    continue; // already added via RB
                }
                seen.Add(tr);
                if (existing.TryGetValue(tr, out var co))
                {
                    newList.Add(co);
                }
                else
                {
                    newList.Add(new ChildOverride { Child = tr, Include = true, SubType = tr.name });
                }
            }
            group.Children = newList;
        }

        internal RootGroup GetRootGroupAt(int index)
        {
            if (m_RootGroups == null) return null;
            if (index < 0 || index >= m_RootGroups.Count) return null;
            return m_RootGroups[index];
        }

        // ---- Canonical label resolution -------------------------------------
        // Update() and BuildCategoryPlan() MUST agree on both the label of an
        // entity and on whether that entity is emitted at all. These helpers are
        // the single source of truth for both passes.

        private static string TypeLabel(RootGroup group)
        {
            return group.Type ?? string.Empty;
        }

        private static string RootSubTypeLabel(RootGroup group)
        {
            return string.IsNullOrEmpty(group.RootSubType) ? group.Root.name : group.RootSubType;
        }

        private static string ChildSubTypeLabel(ChildOverride child)
        {
            return string.IsNullOrEmpty(child.SubType) ? child.Child.name : child.SubType;
        }

        /// <summary>
        /// Resolves a label against the frozen plan. Labels that appear only at
        /// runtime (e.g. an object spawned into a group after CreateSensors) have
        /// no reserved one-hot slot, so they fall back to index 0 rather than
        /// widening the row past the allocated sensor.
        /// </summary>
        private int LookupCategory(Dictionary<string, int> index, string label, string kind)
        {
            int id;
            if (index.TryGetValue(label, out id))
            {
                return id;
            }
            if (!m_WarnedUnknownLabel)
            {
                m_WarnedUnknownLabel = true;
                Debug.LogWarning(
                    $"[EGNNSensor:{m_SensorName}] Unknown {kind} label '{label}' encountered at runtime; " +
                    "it has no reserved one-hot slot and will be reported as index 0. Declare every " +
                    "label in the sensor's root groups before play so a slot is allocated for it.");
            }
            return 0;
        }

        /// <summary>
        /// Freezes label -> one-hot index maps and the resulting caps. Enumerates
        /// entities under exactly the same rules Update() uses, so the row width
        /// computed here is the row width that gets written.
        /// </summary>
        private void BuildCategoryPlan()
        {
            m_TypeIndex.Clear();
            m_SubTypeIndex.Clear();
            if (m_RootGroups != null)
            {
                foreach (var group in m_RootGroups)
                {
                    if (group == null || group.Root == null)
                    {
                        continue;
                    }
                    var typeLabel = TypeLabel(group);
                    if (!m_TypeIndex.ContainsKey(typeLabel))
                    {
                        m_TypeIndex[typeLabel] = m_TypeIndex.Count;
                    }

                    // Root contributes a subtype slot only if it is actually emitted
                    if (group.Root.GetComponent<Rigidbody>() != null || group.Root.GetComponent<Collider>() != null)
                    {
                        var rootLabel = RootSubTypeLabel(group);
                        if (!m_SubTypeIndex.ContainsKey(rootLabel))
                        {
                            m_SubTypeIndex[rootLabel] = m_SubTypeIndex.Count;
                        }
                    }

                    if (!group.IncludeChildren)
                    {
                        continue;
                    }
                    SyncChildrenForGroup(group);
                    foreach (var child in group.Children)
                    {
                        // Include=false entries are skipped by Update(), so they must
                        // not reserve a slot here either.
                        if (child == null || !child.Include || child.Child == null)
                        {
                            continue;
                        }
                        if (child.Child.GetComponent<Rigidbody>() == null && child.Child.GetComponent<Collider>() == null)
                        {
                            continue;
                        }
                        var subLabel = ChildSubTypeLabel(child);
                        if (!m_SubTypeIndex.ContainsKey(subLabel))
                        {
                            m_SubTypeIndex[subLabel] = m_SubTypeIndex.Count;
                        }
                    }
                }
            }
            m_TypeCap = Mathf.Max(1, m_TypeIndex.Count);
            m_SubTypeCap = Mathf.Max(1, m_SubTypeIndex.Count);
            m_PlanBuilt = true;
        }

        private int InferMaxEntities()
        {
            int count = 0;
            if (m_RootGroups == null)
            {
                return 1;
            }
            foreach (var group in m_RootGroups)
            {
                if (group == null || group.Root == null)
                {
                    continue;
                }
                var rootRb = group.Root.GetComponent<Rigidbody>();
                var rootCol = group.Root.GetComponent<Collider>();
                if (rootRb != null || rootCol != null)
                {
                    count++;
                }
                if (group.IncludeChildren && group.Children != null)
                {
                    foreach (var child in group.Children)
                    {
                        if (child == null || !child.Include || child.Child == null)
                        {
                            continue;
                        }
                        var childRb = child.Child.GetComponent<Rigidbody>();
                        var childCol = child.Child.GetComponent<Collider>();
                        if (childRb != null || childCol != null)
                        {
                            count++;
                        }
                    }
                }
            }
            return Mathf.Max(1, count);
        }
    }
}


