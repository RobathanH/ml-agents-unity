using UnityEditor;
using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

namespace Unity.MLAgents.Editor
{
    [CustomEditor(typeof(EGNNSensorComponent))]
    [CanEditMultipleObjects]
    internal class EGNNSensorComponentEditor : UnityEditor.Editor
    {
        private SerializedProperty _sensorName;
        private SerializedProperty _includeRotation;
        private SerializedProperty _includeLinearVelocity;
        private SerializedProperty _includeAngularVelocity;
        private SerializedProperty _virtualRoot;
        private SerializedProperty _rootGroups;
        private bool _showGroups = true;

        private void OnEnable()
        {
            _sensorName = serializedObject.FindProperty("m_SensorName");
            // Expose type/subtype categorical limits
            _includeRotation = serializedObject.FindProperty("m_IncludeRotation");
            _includeLinearVelocity = serializedObject.FindProperty("m_IncludeLinearVelocity");
            _includeAngularVelocity = serializedObject.FindProperty("m_IncludeAngularVelocity");
            _includeRotation = serializedObject.FindProperty("m_IncludeRotation");
            _includeLinearVelocity = serializedObject.FindProperty("m_IncludeLinearVelocity");
            _includeAngularVelocity = serializedObject.FindProperty("m_IncludeAngularVelocity");
            _virtualRoot = serializedObject.FindProperty("m_VirtualRoot");
            _rootGroups = serializedObject.FindProperty("m_RootGroups");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.PropertyField(_sensorName);
            EditorGUILayout.PropertyField(_includeRotation);
            EditorGUILayout.PropertyField(_includeLinearVelocity);
            EditorGUILayout.PropertyField(_includeAngularVelocity);
            EditorGUILayout.PropertyField(_virtualRoot);
            // Draw groups with child listings
            _showGroups = EditorGUILayout.Foldout(_showGroups, "Root Groups", true);
            if (_showGroups)
            {
                var groupsToSync = new List<int>();
                for (int i = 0; i < _rootGroups.arraySize; i++)
                {
                    var groupProp = _rootGroups.GetArrayElementAtIndex(i);
                    EditorGUILayout.BeginVertical("box");
                    // Track changes to Root and IncludeChildren to trigger sync after apply
                    var rootProp = groupProp.FindPropertyRelative("Root");
                    EditorGUI.BeginChangeCheck();
                    EditorGUILayout.PropertyField(rootProp);
                    bool rootChanged = EditorGUI.EndChangeCheck();
                    EditorGUILayout.PropertyField(groupProp.FindPropertyRelative("Type"));
                    EditorGUILayout.PropertyField(groupProp.FindPropertyRelative("RootSubType"));
                    var includeChildrenProp = groupProp.FindPropertyRelative("IncludeChildren");
                    EditorGUI.BeginChangeCheck();
                    EditorGUILayout.PropertyField(includeChildrenProp);
                    bool includeChanged = EditorGUI.EndChangeCheck();
                    if (rootChanged || includeChanged)
                    {
                        groupsToSync.Add(i);
                    }

                    var childrenProp = groupProp.FindPropertyRelative("Children");
                    // Always draw an expanded, read/write list of children when IncludeChildren is true
                    if (includeChildrenProp.boolValue)
                    {
                        EditorGUILayout.LabelField("Children");
                        EditorGUI.indentLevel++;
                        if (childrenProp != null)
                        {
                            for (int c = 0; c < childrenProp.arraySize; c++)
                            {
                                var childProp = childrenProp.GetArrayElementAtIndex(c);
                                EditorGUILayout.BeginHorizontal();
                                EditorGUILayout.PropertyField(childProp.FindPropertyRelative("Include"), GUIContent.none, GUILayout.Width(20));
                                EditorGUILayout.PropertyField(childProp.FindPropertyRelative("Child"), GUIContent.none);
                                EditorGUILayout.PropertyField(childProp.FindPropertyRelative("SubType"), GUIContent.none);
                                EditorGUILayout.EndHorizontal();
                            }
                        }
                        EditorGUI.indentLevel--;
                    }

                    EditorGUILayout.EndVertical();
                }
                if (GUILayout.Button("Add Root Group"))
                {
                    _rootGroups.InsertArrayElementAtIndex(_rootGroups.arraySize);
                }
            }
            serializedObject.ApplyModifiedProperties();

            // Perform child syncs after properties are applied to avoid invalidating current GUI state
            if (_showGroups && targets != null)
            {
                var comp = target as EGNNSensorComponent;
                if (comp != null)
                {
                    // Build a local copy of indices to sync (avoid capturing from inner scope)
                    // Note: We recompute desired groups to sync by inspecting current serialized state again
                    // to ensure indices are still valid.
                    // For simplicity, sync all groups that have IncludeChildren true and a non-null root
                    for (int i = 0; i < _rootGroups.arraySize; i++)
                    {
                        var group = comp.GetRootGroupAt(i);
                        if (group != null && group.IncludeChildren && group.Root != null)
                        {
                            comp.SyncChildrenForGroup(group);
                        }
                    }
                    EditorUtility.SetDirty(comp);
                }
            }
        }

        private EGNNSensorComponent.RootGroup GetGroupByIndex(EGNNSensorComponent comp, int index)
        {
            if (comp == null) return null;
            return comp.GetRootGroupAt(index);
        }
    }
}


