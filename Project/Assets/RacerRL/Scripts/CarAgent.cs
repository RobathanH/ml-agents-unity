using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class CarAgent : Agent
{
    // Static Parameters
    [Header("Drive Force")]
    [Range(0, 10000)]
    public float DriveForce = 100;



    // Control Setup
    [Header("Body Parts")][Space(10)]
    public Transform wheel0;
    public Transform wheel1;
    public Transform wheel2;
    public Transform wheel3;

    // Store most recent action (throttle/desired-rotation-speed)
    private float m_LatestThrottle = 0;

    public override void Initialize()
    {
        updateWheel(wheel0);
        updateWheel(wheel1);
        updateWheel(wheel2);
        updateWheel(wheel3);
    }


    public override void OnEpisodeBegin() { }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(1);
    }


    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var cActions = actionBuffers.ContinuousActions;
        var i = 0;
        
        var throttle = cActions[i++];
        m_LatestThrottle = throttle;
        updateWheel(wheel0);
        updateWheel(wheel1);
        updateWheel(wheel2);
        updateWheel(wheel3);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cActions = actionsOut.ContinuousActions;
        var i = 0;

        var throttle = Input.GetAxis("Vertical") * DriveForce;
        cActions[i++] = throttle;
    }






    private void updateWheel(Transform wheel)
    {
        var rb = wheel.GetComponent<Rigidbody>();
        var hj = wheel.GetComponent<HingeJoint>();

        // Update motor params
        var motor = hj.motor;
        motor.force = DriveForce;
        motor.targetVelocity = m_LatestThrottle;

        hj.motor = motor;
    }
}
