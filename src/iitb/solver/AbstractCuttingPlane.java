package iitb.solver;

import java.util.HashSet;
import java.util.Set;

public abstract class AbstractCuttingPlane {
    
    private int numBuckets;
    
    public AbstractCuttingPlane(int numConstraintBuckets) {
        this.numBuckets = numConstraintBuckets;
    }
    
    protected abstract IConstraint constraintGenerator(int bucket, float[] solution);
    
    protected abstract void solveWithConstraints(Set<IConstraint> constraintSet, float[] initial);
    
    public float[] solve(float[] initial) {
        boolean done = false;
        Set<IConstraint> constraintSet = new HashSet<IConstraint>();
        while (!done) {
            boolean added = false;
            for (int i = 0; i < numBuckets; i++) {
                IConstraint constraint = constraintGenerator(i, initial);
                if (constraint != null) {
                    added = true;
                    constraintSet.add(constraint);
                }
            }
            if (added) {
                solveWithConstraints(constraintSet, initial);
            } else {
                done = true;
            }
        }
        return initial;
    }

}