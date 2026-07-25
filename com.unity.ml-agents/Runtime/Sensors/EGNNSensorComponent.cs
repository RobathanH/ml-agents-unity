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
        /// Whether to include the displacement from the emitted node position to
        /// the entity's centre. Only meaningful for sources that place the node
        /// somewhere other than the centre (see <see cref="EGNNEntity"/>); it is
        /// zero for hierarchy-derived entities. Emitted as an equivariant vector
        /// channel, so the encoder's has_center_offset must match this.
        /// </summary>
        [SerializeField]
        private bool m_IncludeCenterOffset = false;

        /// <summary>
        /// Whether to include per-entity half-extents. These are lengths along the
        /// entity's own axes, hence rotation-invariant, and are consumed as plain
        /// scalars. With position, rotation and centre offset they make a box's
        /// encoding lossless.
        /// </summary>
        [SerializeField]
        private bool m_IncludeExtent = false;

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

        /// <summary>
        /// Components implementing <see cref="IEGNNEntitySource"/>, for entities
        /// that cannot be discovered from a static hierarchy (procedurally
        /// spawned obstacles and the like). Each source reserves its declared
        /// entity budget and vocabulary in the frozen plan.
        /// </summary>
        [SerializeField]
        private List<MonoBehaviour> m_EntitySources = new List<MonoBehaviour>();

        private BufferSensor m_Sensor;

        // Reused across steps. Update() runs at frame rate for every agent, so
        // allocating a row list per entity per frame was measurable GC churn even
        // at CrawlerSumo's 19 entities; parkour buffers are several times larger.
        private readonly List<EGNNEntity> m_Entities = new List<EGNNEntity>();
        private float[] m_RowBuffer;

        // Per-entity feature size: 3 pos + optional attrs + one-hot(type) + one-hot(subtype)
        private int ComputeObservableSize()
        {
            int size = 3; // position
            if (m_IncludeRotation) size += 4;
            if (m_IncludeLinearVelocity) size += 3;
            if (m_IncludeAngularVelocity) size += 3;
            if (m_IncludeCenterOffset) size += 3;
            if (m_IncludeExtent) size += 3;
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
            // Infer capacity from roots, included children and source budgets
            m_MaxEntities = InferMaxEntities();
            int obsSize = ComputeObservableSize();
            m_RowBuffer = new float[obsSize];
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
            m_Entities.Clear();

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
                    AddHierarchyEntity(group.Root.transform, rootRb, rootCol, tidRaw, sidRaw);
                }

                if (m_Entities.Count >= m_MaxEntities) break;

                // Children selection. The override list is NOT re-synced here:
                // BuildCategoryPlan already synced it, and the plan is frozen, so
                // a child appearing afterwards has no reserved one-hot slot
                // anyway. Re-syncing every frame only burned allocations.
                if (group.IncludeChildren)
                {
                    foreach (var child in group.Children)
                    {
                        if (!child.Include || child.Child == null) continue;
                        var childRb = child.Child.GetComponent<Rigidbody>();
                        var childCol = child.Child.GetComponent<Collider>();
                        if (childRb == null && childCol == null) continue;
                        int sidRaw2 = LookupCategory(m_SubTypeIndex, ChildSubTypeLabel(child), "subtype");
                        AddHierarchyEntity(child.Child, childRb, childCol, tidRaw, sidRaw2);
                        if (m_Entities.Count >= m_MaxEntities) break;
                    }
                }
                if (m_Entities.Count >= m_MaxEntities) break;
            }

            // Dynamic sources. Each is capped at the budget it declared, so the
            // total can never exceed what CreateSensors allocated.
            if (m_EntitySources != null)
            {
                foreach (var behaviour in m_EntitySources)
                {
                    var source = behaviour as IEGNNEntitySource;
                    if (source == null) continue;
                    int budget = Mathf.Min(source.MaxEntities, m_MaxEntities - m_Entities.Count);
                    if (budget <= 0) break;
                    int before = m_Entities.Count;
                    source.CollectEntities(m_Entities, budget);
                    // A source that overruns its budget would silently shift every
                    // later entity's row, so truncate rather than trust it.
                    if (m_Entities.Count - before > budget)
                    {
                        m_Entities.RemoveRange(before + budget, m_Entities.Count - before - budget);
                    }
                }
            }

            // Coordinate transform to virtual root if provided
            if (m_VirtualRoot != null)
            {
                var invRot = Quaternion.Inverse(m_VirtualRoot.transform.rotation);
                var origin = m_VirtualRoot.transform.position;
                for (int i = 0; i < m_Entities.Count; i++)
                {
                    var e = m_Entities[i];
                    e.Position = invRot * (e.Position - origin);
                    e.LinearVelocity = invRot * e.LinearVelocity;
                    e.AngularVelocity = invRot * e.AngularVelocity;
                    // A displacement, so it rotates but does not translate.
                    e.CenterOffset = invRot * e.CenterOffset;
                    e.Rotation = invRot * e.Rotation;
                    // Extent is measured along the entity's own axes and is
                    // therefore unaffected by a change of reference frame.
                    m_Entities[i] = e;
                }
            }

            // Write to buffer sensor as fixed-size rows; remaining rows stay zero as
            // padding. Caps are NOT recomputed here -- they are frozen in
            // BuildCategoryPlan so the row width matches the allocated sensor.
            for (int i = 0; i < m_Entities.Count; i++)
            {
                m_Sensor.AppendObservation(BuildRow(m_Entities[i]));
            }
        }

        private void AddHierarchyEntity(Transform tr, Rigidbody rb, Collider col, int typeId, int subTypeId)
        {
            var e = new EGNNEntity
            {
                Position = rb != null ? rb.position : tr.position,
                Rotation = rb != null ? rb.rotation : tr.rotation,
                LinearVelocity = rb != null ? rb.linearVelocity : Vector3.zero,
                AngularVelocity = rb != null ? rb.angularVelocity : Vector3.zero,
                // Hierarchy entities put their node at their own origin, so the
                // centre offset is zero by construction.
                CenterOffset = Vector3.zero,
                Extent = m_IncludeExtent ? LocalHalfExtents(col) : Vector3.zero,
                TypeId = typeId,
                SubTypeId = subTypeId,
            };
            m_Entities.Add(e);
        }

        /// <summary>
        /// Half-extents along the collider's own axes. Deliberately not
        /// <c>Collider.bounds</c>, which is a world-space AABB and therefore
        /// changes with orientation -- that would not be rotation-invariant and
        /// could not be fed to the encoder as a scalar.
        /// </summary>
        internal static Vector3 LocalHalfExtents(Collider col)
        {
            if (col == null)
            {
                return Vector3.zero;
            }
            var s = col.transform.lossyScale;
            var abs = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));

            var box = col as BoxCollider;
            if (box != null)
            {
                return Vector3.Scale(box.size * 0.5f, abs);
            }
            var sphere = col as SphereCollider;
            if (sphere != null)
            {
                // Unity scales a sphere collider by the largest axis scale.
                float r = sphere.radius * Mathf.Max(abs.x, Mathf.Max(abs.y, abs.z));
                return new Vector3(r, r, r);
            }
            var capsule = col as CapsuleCollider;
            if (capsule != null)
            {
                // Radius follows the larger of the two off-axis scales; height
                // follows the direction axis. Matches Unity's own convention.
                float axisScale, radScale;
                switch (capsule.direction)
                {
                    case 0: axisScale = abs.x; radScale = Mathf.Max(abs.y, abs.z); break;
                    case 2: axisScale = abs.z; radScale = Mathf.Max(abs.x, abs.y); break;
                    default: axisScale = abs.y; radScale = Mathf.Max(abs.x, abs.z); break;
                }
                float r = capsule.radius * radScale;
                float half = Mathf.Max(capsule.height * 0.5f * axisScale, r);
                switch (capsule.direction)
                {
                    case 0: return new Vector3(half, r, r);
                    case 2: return new Vector3(r, r, half);
                    default: return new Vector3(r, half, r);
                }
            }
            var mesh = col as MeshCollider;
            if (mesh != null && mesh.sharedMesh != null)
            {
                return Vector3.Scale(mesh.sharedMesh.bounds.extents, abs);
            }
            return Vector3.zero;
        }

        private float[] BuildRow(EGNNEntity e)
        {
            var obs = m_RowBuffer;
            int n = 0;
            // Position first (for EGNN positional dims)
            obs[n++] = e.Position.x; obs[n++] = e.Position.y; obs[n++] = e.Position.z;
            if (m_IncludeRotation)
            {
                obs[n++] = e.Rotation.x; obs[n++] = e.Rotation.y;
                obs[n++] = e.Rotation.z; obs[n++] = e.Rotation.w;
            }
            if (m_IncludeLinearVelocity)
            {
                obs[n++] = e.LinearVelocity.x; obs[n++] = e.LinearVelocity.y; obs[n++] = e.LinearVelocity.z;
            }
            if (m_IncludeAngularVelocity)
            {
                obs[n++] = e.AngularVelocity.x; obs[n++] = e.AngularVelocity.y; obs[n++] = e.AngularVelocity.z;
            }
            // Equivariant vector channels must precede the scalar block: the
            // encoder derives its scalar slice as "everything after the last
            // declared vector field".
            if (m_IncludeCenterOffset)
            {
                obs[n++] = e.CenterOffset.x; obs[n++] = e.CenterOffset.y; obs[n++] = e.CenterOffset.z;
            }
            if (m_IncludeExtent)
            {
                obs[n++] = e.Extent.x; obs[n++] = e.Extent.y; obs[n++] = e.Extent.z;
            }
            // Type one-hot (fixed size)
            int typeSize = Math.Max(1, m_TypeCap);
            int tid = Mathf.Clamp(e.TypeId, 0, typeSize - 1);
            for (int i = 0; i < typeSize; i++)
            {
                obs[n++] = i == tid ? 1f : 0f;
            }
            // SubType one-hot (fixed size)
            int subSize = Math.Max(1, m_SubTypeCap);
            int sid = Mathf.Clamp(e.SubTypeId, 0, subSize - 1);
            for (int i = 0; i < subSize; i++)
            {
                obs[n++] = i == sid ? 1f : 0f;
            }
            return obs;
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

            // Dynamic sources reserve their whole declared vocabulary, whether or
            // not they happen to emit it this episode. Their membership changes
            // between episodes, so anything resolved lazily would give the same
            // label a different one-hot slot from one reset to the next.
            if (m_EntitySources != null)
            {
                foreach (var behaviour in m_EntitySources)
                {
                    var source = behaviour as IEGNNEntitySource;
                    if (source == null)
                    {
                        if (behaviour != null)
                        {
                            Debug.LogWarning(
                                $"[EGNNSensor:{m_SensorName}] Entity source '{behaviour.name}' " +
                                $"({behaviour.GetType().Name}) does not implement IEGNNEntitySource " +
                                "and will be ignored.");
                        }
                        continue;
                    }
                    var typeLabel = source.TypeLabel ?? string.Empty;
                    if (!m_TypeIndex.ContainsKey(typeLabel))
                    {
                        m_TypeIndex[typeLabel] = m_TypeIndex.Count;
                    }
                    var subLabels = source.SubTypeLabels ?? Array.Empty<string>();
                    foreach (var label in subLabels)
                    {
                        var key = label ?? string.Empty;
                        if (!m_SubTypeIndex.ContainsKey(key))
                        {
                            m_SubTypeIndex[key] = m_SubTypeIndex.Count;
                        }
                    }
                }
            }

            m_TypeCap = Mathf.Max(1, m_TypeIndex.Count);
            m_SubTypeCap = Mathf.Max(1, m_SubTypeIndex.Count);
            m_PlanBuilt = true;

            // Hand each source its resolved indices now that the plan is final, so
            // it can emit them directly instead of hashing strings every step.
            if (m_EntitySources != null)
            {
                foreach (var behaviour in m_EntitySources)
                {
                    var source = behaviour as IEGNNEntitySource;
                    if (source == null) continue;
                    var subLabels = source.SubTypeLabels ?? Array.Empty<string>();
                    var resolved = new int[subLabels.Length];
                    for (int i = 0; i < subLabels.Length; i++)
                    {
                        resolved[i] = m_SubTypeIndex[subLabels[i] ?? string.Empty];
                    }
                    source.BindCategoryIndices(m_TypeIndex[source.TypeLabel ?? string.Empty], resolved);
                }
            }
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
            if (m_EntitySources != null)
            {
                foreach (var behaviour in m_EntitySources)
                {
                    var source = behaviour as IEGNNEntitySource;
                    if (source != null)
                    {
                        count += Mathf.Max(0, source.MaxEntities);
                    }
                }
            }
            return Mathf.Max(1, count);
        }
    }
}


