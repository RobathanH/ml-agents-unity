using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Feeds the nearest track boxes to the EGNN sensor as surface-contact nodes.
    /// </summary>
    /// <remarks>
    /// Each obstacle is emitted at the closest point on its surface to the agent's
    /// body, not at its centroid. For a 12m floor tile or a track-spanning wall the
    /// centroid is metres from anything the agent can touch, and the EGNN's radial
    /// prior -- kNN selection and the distance term in the message MLP -- would then
    /// be keyed on a distance uncorrelated with the one that matters. Reporting the
    /// centre offset alongside keeps the encoding lossless: position + offset gives
    /// the centre back, and with rotation and half-extents that fully determines the
    /// box, so every surface query the agent needs stays computable in principle.
    ///
    /// Obstacles are static within an episode, so nothing here needs velocities.
    /// </remarks>
    public class ParkourObstacleSource : MonoBehaviour, IEGNNEntitySource
    {
        public ParkourTrackGenerator Track;

        [Tooltip("Point the closest-surface query is taken from -- the agent's body.")]
        public Transform Reference;

        [Tooltip("Entity budget reserved in the sensor. Unused slots pad with zeros.")]
        public int MaxObstacles = 16;

        [Tooltip("Longitudinal window around the agent. Boxes outside it are skipped "
                 + "before any distance work.")]
        public float LookAhead = 20f;
        public float LookBehind = 6f;

        // Geometric and kinematic only -- deliberately no wall/ramp/ledge labels.
        // Those roles are recoverable from the geometry, and a one-hot for them
        // would let the policy key a reflex on the label instead of looking at the
        // shape, which is exactly what stops the behaviour generalising to box
        // configurations that were never trained on. "dynamic" earns its slot
        // because "can this be pushed" is a mass property, not a shape.
        private static readonly string[] k_SubTypes = { "static", "dynamic" };

        private int m_TypeIndex;
        private int[] m_SubIndices = { 0, 0 };

        // Reused every step; CollectEntities runs at frame rate per agent.
        private readonly List<ParkourBox> m_Candidates = new List<ParkourBox>();
        private readonly List<float> m_Distances = new List<float>();
        private readonly List<Vector3> m_ClosestPoints = new List<Vector3>();

        public string TypeLabel => "obstacle";
        public string[] SubTypeLabels => k_SubTypes;
        public int MaxEntities => MaxObstacles;

        public void BindCategoryIndices(int typeIndex, int[] subTypeIndices)
        {
            m_TypeIndex = typeIndex;
            if (subTypeIndices != null && subTypeIndices.Length == k_SubTypes.Length)
            {
                m_SubIndices = subTypeIndices;
            }
        }

        public void CollectEntities(List<EGNNEntity> into, int budget)
        {
            if (Track == null || Reference == null || budget <= 0) return;

            var p = Reference.position;
            m_Candidates.Clear();
            m_Distances.Clear();
            m_ClosestPoints.Clear();

            var boxes = Track.ActiveBoxes;
            for (int i = 0; i < boxes.Count; i++)
            {
                var b = boxes[i];
                // Longitudinal window first: rejects most of the track without
                // touching the closest-point maths.
                if (b.WorldMax.z < p.z - LookBehind || b.WorldMin.z > p.z + LookAhead) continue;
                var q = b.ClosestPoint(p);
                m_Candidates.Add(b);
                m_ClosestPoints.Add(q);
                m_Distances.Add((q - p).sqrMagnitude);
            }

            // Partial selection for the k nearest. With a few dozen candidates and
            // a budget of ~16 this beats sorting, and it allocates nothing.
            int n = m_Candidates.Count;
            int take = Mathf.Min(budget, n);
            for (int k = 0; k < take; k++)
            {
                int best = k;
                for (int j = k + 1; j < n; j++)
                {
                    if (m_Distances[j] < m_Distances[best]) best = j;
                }
                if (best != k)
                {
                    (m_Candidates[k], m_Candidates[best]) = (m_Candidates[best], m_Candidates[k]);
                    (m_Distances[k], m_Distances[best]) = (m_Distances[best], m_Distances[k]);
                    (m_ClosestPoints[k], m_ClosestPoints[best]) = (m_ClosestPoints[best], m_ClosestPoints[k]);
                }

                var b = m_Candidates[k];
                var q = m_ClosestPoints[k];
                into.Add(new EGNNEntity
                {
                    Position = q,
                    Rotation = b.Rotation,
                    LinearVelocity = Vector3.zero,
                    AngularVelocity = Vector3.zero,
                    // q + CenterOffset == b.Center, so the centre survives the
                    // move to the surface and the box stays fully determined.
                    CenterOffset = b.Center - q,
                    Extent = b.HalfExtents,
                    TypeId = m_TypeIndex,
                    SubTypeId = m_SubIndices[b.IsDynamic ? 1 : 0],
                });
            }
        }
    }
}
