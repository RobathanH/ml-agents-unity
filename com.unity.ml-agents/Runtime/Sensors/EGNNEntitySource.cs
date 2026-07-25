using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// One row of an <see cref="EGNNSensorComponent"/> observation, before the
    /// virtual-root transform and one-hot expansion are applied.
    /// </summary>
    /// <remarks>
    /// <see cref="Position"/> is not required to be the entity's centre of mass.
    /// For large obstacles the centroid is a poor summary -- a long wall's centre
    /// can be many metres away while its face is within reach -- and the EGNN's
    /// radial prior (kNN selection and the distance term in the message MLP) then
    /// keys on a distance that is uncorrelated with the one that matters. A source
    /// may instead emit the closest point on the entity's surface and report
    /// <see cref="CenterOffset"/> so that <c>Position + CenterOffset</c> recovers
    /// the centre. Together with <see cref="Rotation"/> and <see cref="Extent"/>
    /// that is a lossless encoding of a box, so nothing is discarded by the
    /// re-parameterisation -- it only moves the node to where the interaction is.
    /// </remarks>
    public struct EGNNEntity
    {
        /// <summary>World-space node position. See the remarks on this struct.</summary>
        public Vector3 Position;

        /// <summary>World-space orientation of the entity's own frame.</summary>
        public Quaternion Rotation;

        public Vector3 LinearVelocity;
        public Vector3 AngularVelocity;

        /// <summary>
        /// Displacement from <see cref="Position"/> to the entity's reference
        /// centre. Zero when <see cref="Position"/> already is the centre. This is
        /// a displacement, so it rotates with the frame and is emitted as an
        /// equivariant vector channel.
        /// </summary>
        public Vector3 CenterOffset;

        /// <summary>
        /// Half-extents in the entity's own frame. These are lengths measured
        /// along the entity's own axes, so they are invariant under global
        /// rotation and are emitted as plain scalars.
        /// </summary>
        public Vector3 Extent;

        /// <summary>Index into the sensor's frozen type vocabulary.</summary>
        public int TypeId;

        /// <summary>Index into the sensor's frozen subtype vocabulary.</summary>
        public int SubTypeId;
    }

    /// <summary>
    /// Supplies entities that are not discoverable from a static transform
    /// hierarchy -- procedurally spawned obstacles, pooled objects, or any set
    /// whose membership changes between episodes.
    /// </summary>
    /// <remarks>
    /// The sensor freezes its one-hot vocabulary and row width at
    /// <c>CreateSensors</c>, so a source must declare <see cref="TypeLabel"/>,
    /// <see cref="SubTypeLabels"/> and <see cref="MaxEntities"/> up front and
    /// never exceed them. In exchange the source is handed its resolved category
    /// indices once via <see cref="BindCategoryIndices"/> and can emit them
    /// directly each step, with no per-step string hashing.
    /// </remarks>
    public interface IEGNNEntitySource
    {
        /// <summary>Type label shared by every entity this source emits.</summary>
        string TypeLabel { get; }

        /// <summary>
        /// Every subtype label this source may emit, in the order its
        /// <c>CollectEntities</c> indices refer to them.
        /// </summary>
        string[] SubTypeLabels { get; }

        /// <summary>
        /// Upper bound on entities emitted per step. The sensor reserves this
        /// many rows; unused rows stay zero and are masked as padding.
        /// </summary>
        int MaxEntities { get; }

        /// <summary>
        /// Called once after the sensor's category plan is frozen.
        /// <paramref name="subTypeIndices"/> is parallel to
        /// <see cref="SubTypeLabels"/> and maps local index -> global one-hot slot.
        /// </summary>
        void BindCategoryIndices(int typeIndex, int[] subTypeIndices);

        /// <summary>
        /// Appends at most <paramref name="budget"/> entities to
        /// <paramref name="into"/>. Called every observation step, so
        /// implementations should avoid allocating.
        /// </summary>
        void CollectEntities(List<EGNNEntity> into, int budget);
    }
}
