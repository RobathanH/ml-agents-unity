using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// The environment's only geometric primitive: an oriented box. Floor tiles,
    /// walls, ramps, ledges, pebbles and overhangs are all parameterisations of
    /// this, which is what lets the whole obstacle vocabulary share one EGNN node
    /// type instead of minting a one-hot per semantic role.
    /// </summary>
    /// <remarks>
    /// World-space axes and half-extents are cached at placement time. Obstacles
    /// are static for the whole episode, so recomputing them from the transform on
    /// every sensor query would be pure waste -- and the sensor queries this at
    /// frame rate for every agent.
    /// </remarks>
    public class ParkourBox
    {
        public Transform Tr;
        public Vector3 Center;
        public Quaternion Rotation;
        public Vector3 HalfExtents;

        /// <summary>Movable boxes get a distinct subtype: "can this be pushed" is a
        /// mass property and is genuinely not inferable from shape.</summary>
        public bool IsDynamic;

        /// <summary>Set for boxes that form walkable ground, so the feasibility
        /// pass can tell a floor tile from something standing on it.</summary>
        public bool IsFloor;

        /// <summary>World AABB, for broad-phase rejection only. Never use it for
        /// surface queries -- see the remarks on <see cref="VerticalSpanAt"/>.
        /// </summary>
        public Vector3 WorldMin, WorldMax;

        private Vector3 m_Right, m_Up, m_Fwd;

        public void Place(Vector3 center, Quaternion rotation, Vector3 halfExtents)
        {
            Center = center;
            Rotation = rotation;
            HalfExtents = halfExtents;
            m_Right = rotation * Vector3.right;
            m_Up = rotation * Vector3.up;
            m_Fwd = rotation * Vector3.forward;
            var e = new Vector3(
                Mathf.Abs(m_Right.x) * halfExtents.x + Mathf.Abs(m_Up.x) * halfExtents.y + Mathf.Abs(m_Fwd.x) * halfExtents.z,
                Mathf.Abs(m_Right.y) * halfExtents.x + Mathf.Abs(m_Up.y) * halfExtents.y + Mathf.Abs(m_Fwd.y) * halfExtents.z,
                Mathf.Abs(m_Right.z) * halfExtents.x + Mathf.Abs(m_Up.z) * halfExtents.y + Mathf.Abs(m_Fwd.z) * halfExtents.z);
            WorldMin = center - e;
            WorldMax = center + e;
            if (Tr != null)
            {
                Tr.SetPositionAndRotation(center, rotation);
                Tr.localScale = halfExtents * 2f;
            }
        }

        /// <summary>
        /// Closest point on the box surface to <paramref name="p"/>, as a clamp in
        /// the box's own frame.
        /// </summary>
        /// <remarks>
        /// Deliberately not Collider.ClosestPoint: that is a physics query, and
        /// this runs (obstacles x agents x frame rate) times. For a box the
        /// analytic form is exact and allocation-free.
        ///
        /// When p is inside the box the clamp is a no-op and the result is p
        /// itself. That is the correct reading -- "you are penetrating this" -- and
        /// the centre is still recoverable through the reported centre offset.
        /// </remarks>
        public Vector3 ClosestPoint(Vector3 p)
        {
            Vector3 d = p - Center;
            float x = Mathf.Clamp(Vector3.Dot(d, m_Right), -HalfExtents.x, HalfExtents.x);
            float y = Mathf.Clamp(Vector3.Dot(d, m_Up), -HalfExtents.y, HalfExtents.y);
            float z = Mathf.Clamp(Vector3.Dot(d, m_Fwd), -HalfExtents.z, HalfExtents.z);
            return Center + m_Right * x + m_Up * y + m_Fwd * z;
        }

        /// <summary>
        /// Vertical span of the box over the vertical line through
        /// <paramref name="x"/>,<paramref name="z"/>, or false if it does not
        /// cover that column. Used by the feasibility rasteriser, which needs
        /// "what is the top/bottom surface here" rather than a distance.
        /// </summary>
        /// <remarks>
        /// Exact: intersects the vertical line through (x, z) with the oriented
        /// box by the slab method, in the box's own frame.
        ///
        /// A world AABB is emphatically not good enough here. A ramp is a wide,
        /// thin, rotated slab, and its AABB is a tall block -- an 8.5m-wide slab
        /// tilted 20 degrees has an AABB 1.45m taller than the surface it
        /// represents. The feasibility rasteriser would read that as an
        /// unclimbable wall and reject layouts that are in fact gentle slopes.
        ///
        /// Because the ray origin is at y = 0 and the direction is world up, the
        /// intersection parameters are world y coordinates directly.
        /// </remarks>
        public bool VerticalSpanAt(float x, float z, out float bottom, out float top)
        {
            bottom = 0f;
            top = 0f;
            Vector3 o = new Vector3(x, 0f, z) - Center;
            float tmin = float.NegativeInfinity;
            float tmax = float.PositiveInfinity;
            // World up, resolved onto each box axis.
            if (!Slab(Vector3.Dot(o, m_Right), m_Right.y, HalfExtents.x, ref tmin, ref tmax)) return false;
            if (!Slab(Vector3.Dot(o, m_Up), m_Up.y, HalfExtents.y, ref tmin, ref tmax)) return false;
            if (!Slab(Vector3.Dot(o, m_Fwd), m_Fwd.y, HalfExtents.z, ref tmin, ref tmax)) return false;
            if (tmax < tmin) return false;
            bottom = tmin;
            top = tmax;
            return true;
        }

        private static bool Slab(float o, float d, float h, ref float tmin, ref float tmax)
        {
            if (Mathf.Abs(d) < 1e-6f)
            {
                // Ray parallel to this slab: either always inside it or never.
                return Mathf.Abs(o) <= h;
            }
            float t1 = (-h - o) / d;
            float t2 = (h - o) / d;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            return tmin <= tmax;
        }
    }
}
