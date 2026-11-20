# Source this in every container terminal:   source ~/dog-codes/frontier-fastlio/dog_tools/scripts/env.sh
# (it must be SOURCED, not run with "bash env.sh", otherwise nothing stays set in your terminal)
# Sets up ROS 2 + the TA workspaces and prints what it found. It does not set ROS_DOMAIN_ID.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "env.sh must be sourced:  source ~/dog-codes/frontier-fastlio/dog_tools/scripts/env.sh"
  exit 1
fi

for f in /opt/ros/humble/setup.bash ~/lite3_ws/install/setup.bash ~/ros2_ws/install/setup.bash; do
  if [ -f "$f" ]; then
    source "$f"
    echo "sourced: $f"
  else
    echo "MISSING: $f"
  fi
done

if [ -z "$ROS_DOMAIN_ID" ]; then
  echo "WARNING: ROS_DOMAIN_ID is not set. Set it to the dog's domain (same in every terminal), e.g.  export ROS_DOMAIN_ID=1"
else
  echo "ROS_DOMAIN_ID = $ROS_DOMAIN_ID   (must be the domain where 'ros2 topic list' shows /odom)"
fi
